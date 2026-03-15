#include <stdint.h>
#include <stdio.h>

#include "mnist_q7_1_data.h"
#include "svm_cpu_quantized.h"
#include "svm_ps_driver.h"
#include "xil_cache.h"
#include "xil_cache_l.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xtime_l.h"

#ifndef XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#ifndef XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

/* Match TB acceptance rule. */
#define TB_MIN_ACCURACY 0.98f

#define TOTAL_IMAGES ((uint16_t)MNIST_NUM_IMAGES)
#define SPLIT_SCAN_MAX_CANDIDATES 16u
#define DMA_ADDR_ALIGN_BYTES 8u
#define SPLIT_BALANCE_WINDOW_PCT 5u

/*
 * Deployment mode:
 * - 1: fixed split, one-shot parallel inference (no online scan)
 * - 0: keep online split exploration/tuning flow
 */
#define ENABLE_DEPLOY_MODE 1u
#define DEPLOY_FIXED_N_PS 0u    /* fastest end-to-end: all images on PL */
#define DEPLOY_REPORT_SPLIT_SERIAL 0u

static uint8_t g_pl_predictions[TOTAL_IMAGES] __attribute__((aligned(64)));
static uint8_t g_cpu_predictions[TOTAL_IMAGES] __attribute__((aligned(64)));
static uint8_t g_parallel_predictions[TOTAL_IMAGES] __attribute__((aligned(64)));

typedef struct {
    uint16_t n_ps;
    uint16_t n_pl;
    uint64_t total_cycles;
    uint64_t ps_cycles;
    uint64_t pl_cycles;
} split_run_result_t;

static inline uint64_t cycles_to_us(uint64_t cycles) {
    return (cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
}

static inline uint64_t u64_abs_diff(uint64_t a, uint64_t b) {
    return (a > b) ? (a - b) : (b - a);
}

static uint16_t align_n_ps_for_dma(uint16_t n_ps) {
    uint16_t aligned;

    if (n_ps < DMA_ADDR_ALIGN_BYTES) {
        return DMA_ADDR_ALIGN_BYTES;
    }

    aligned = (uint16_t)(((uint32_t)n_ps + (DMA_ADDR_ALIGN_BYTES / 2u)) &
                         (uint32_t)~(DMA_ADDR_ALIGN_BYTES - 1u));
    if (aligned < DMA_ADDR_ALIGN_BYTES) {
        aligned = DMA_ADDR_ALIGN_BYTES;
    }
    if (aligned >= TOTAL_IMAGES) {
        aligned = (uint16_t)(TOTAL_IMAGES - 1u);
        aligned = (uint16_t)(aligned & (uint16_t)~(DMA_ADDR_ALIGN_BYTES - 1u));
        if (aligned < DMA_ADDR_ALIGN_BYTES) {
            aligned = DMA_ADDR_ALIGN_BYTES;
        }
    }
    return aligned;
}

static int run_parallel_split(uint16_t n_ps, uint8_t *pred_out, split_run_result_t *result) {
    int status = XST_SUCCESS;
    uint16_t n_pl;
    uint64_t pl_dma_cycles = 0u;
    uint64_t pl_kernel_cycles = 0u;
    uint64_t ps_cycles = 0u;
    XTime t_start;
    XTime t_end;

    if ((pred_out == NULL) || (result == NULL)) {
        return XST_INVALID_PARAM;
    }
    if (n_ps > TOTAL_IMAGES) {
        return XST_INVALID_PARAM;
    }
    if ((n_ps % DMA_ADDR_ALIGN_BYTES) != 0u) {
        return XST_INVALID_PARAM;
    }

    n_pl = (uint16_t)(TOTAL_IMAGES - n_ps);

    XTime_GetTime(&t_start);

    if (n_pl > 0u) {
        const int8_t *pl_in = &g_mnist_test_q7_1[(uint32_t)n_ps * MNIST_IMG_SIZE];
        uint8_t *pl_out = &pred_out[n_ps];
        status = svm_run_batch_async_start(pl_in, pl_out, n_pl);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    if (n_ps > 0u) {
        status = svm_cpu_quantized_run_batch_timed(g_mnist_test_q7_1, pred_out, n_ps, &ps_cycles);
        if (status != XST_SUCCESS) {
            if (n_pl > 0u) {
                uint64_t drain_dma = 0u;
                uint64_t drain_kernel = 0u;
                (void)svm_run_batch_async_wait(&pred_out[n_ps], n_pl, &drain_dma, &drain_kernel);
            }
            return status;
        }
    }

    if (n_pl > 0u) {
        status = svm_run_batch_async_wait(&pred_out[n_ps], n_pl, &pl_dma_cycles, &pl_kernel_cycles);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    XTime_GetTime(&t_end);

    result->n_ps = n_ps;
    result->n_pl = n_pl;
    result->total_cycles = (uint64_t)(t_end - t_start);
    result->ps_cycles = ps_cycles;
    result->pl_cycles = (pl_dma_cycles > pl_kernel_cycles) ? pl_dma_cycles : pl_kernel_cycles;
    return XST_SUCCESS;
}

static int run_serial_split(uint16_t n_ps,
                            uint8_t *pred_out,
                            uint64_t *pl_cycles_out,
                            uint64_t *ps_cycles_out,
                            uint64_t *total_cycles_out) {
    int status;
    uint16_t n_pl;
    uint64_t pl_dma_cycles = 0u;
    uint64_t pl_kernel_cycles = 0u;
    uint64_t ps_cycles = 0u;
    XTime t_start;
    XTime t_end;

    if ((pred_out == NULL) || (pl_cycles_out == NULL) || (ps_cycles_out == NULL) || (total_cycles_out == NULL)) {
        return XST_INVALID_PARAM;
    }
    if ((n_ps > TOTAL_IMAGES) || ((n_ps % DMA_ADDR_ALIGN_BYTES) != 0u)) {
        return XST_INVALID_PARAM;
    }

    n_pl = (uint16_t)(TOTAL_IMAGES - n_ps);

    XTime_GetTime(&t_start);

    if (n_pl > 0u) {
        const int8_t *pl_in = &g_mnist_test_q7_1[(uint32_t)n_ps * MNIST_IMG_SIZE];
        uint8_t *pl_out = &pred_out[n_ps];
        status = svm_run_batch_timed(pl_in, pl_out, n_pl, &pl_dma_cycles, &pl_kernel_cycles);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    if (n_ps > 0u) {
        status = svm_cpu_quantized_run_batch_timed(g_mnist_test_q7_1, pred_out, n_ps, &ps_cycles);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    XTime_GetTime(&t_end);

    *pl_cycles_out = (pl_dma_cycles > pl_kernel_cycles) ? pl_dma_cycles : pl_kernel_cycles;
    *ps_cycles_out = ps_cycles;
    *total_cycles_out = (uint64_t)(t_end - t_start);
    return XST_SUCCESS;
}

static uint32_t build_split_candidates(uint16_t center_n_ps, uint16_t *out, uint32_t max_n) {
    static const int16_t offsets[] = {-40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40};
    uint32_t count = 0u;

    if ((out == NULL) || (max_n == 0u)) {
        return 0u;
    }

    for (uint32_t i = 0u; i < (uint32_t)(sizeof(offsets) / sizeof(offsets[0])); ++i) {
        int32_t cand = (int32_t)center_n_ps + (int32_t)offsets[i];
        int duplicated = 0;

        if ((cand < 1) || (cand >= (int32_t)TOTAL_IMAGES)) {
            continue;
        }
        cand = (int32_t)align_n_ps_for_dma((uint16_t)cand);

        for (uint32_t j = 0u; j < count; ++j) {
            if (out[j] == (uint16_t)cand) {
                duplicated = 1;
                break;
            }
        }

        if (!duplicated) {
            out[count++] = (uint16_t)cand;
            if (count >= max_n) {
                return count;
            }
        }
    }

    return count;
}

int main(void) {
    int status;
    uint64_t pl_dma_cycles = 0u;
    uint64_t pl_kernel_cycles = 0u;
    uint64_t pl_cycles;
    uint64_t ps_cycles = 0u;
    double pl_cycles_per_img;
    double ps_cycles_per_img;
    uint16_t n_ps_equal_est;
    uint16_t candidates[SPLIT_SCAN_MAX_CANDIDATES];
    uint32_t candidate_count;
    split_run_result_t valid_runs[SPLIT_SCAN_MAX_CANDIDATES];
    uint32_t valid_count = 0u;
    split_run_result_t best_run;
    int best_valid = 0;
    float final_acc = 0.0f;
    uint32_t final_mis = 0u;
    uint32_t final_acc_x1e6 = 0u;
    int final_acc_ok = 0;

    printf("Starting Program\r\n");
    Xil_ICacheEnable();
    Xil_DCacheEnable();
    Xil_L2CacheEnable();

    status = svm_init_hw();
    if (status != XST_SUCCESS) {
        printf("svm_init_hw failed: %d\r\n", status);
        return status;
    }

    status = svm_cpu_quantized_prepare();
    if (status != XST_SUCCESS) {
        printf("svm_cpu_quantized_prepare failed: %d\r\n", status);
        return status;
    }

#if ENABLE_DEPLOY_MODE
    {
        split_run_result_t run;
        uint16_t n_ps = (uint16_t)DEPLOY_FIXED_N_PS;
        uint64_t wall_us;
        uint64_t makespan_us;
        uint64_t ps_us;
        uint64_t pl_us;
        uint64_t gap_us;
        uint64_t makespan_ips_x1000 = 0u;
        uint64_t ps_share_x1000;

#if DEPLOY_REPORT_SPLIT_SERIAL
        uint64_t serial_total_cycles = 0u;
        uint64_t serial_ps_cycles = 0u;
        uint64_t serial_pl_cycles = 0u;
        uint64_t speedup_vs_split_serial_x1000 = 0u;
#endif

        if (n_ps > 0u) {
            n_ps = align_n_ps_for_dma(n_ps);
        }
        if (n_ps >= TOTAL_IMAGES) {
            n_ps = align_n_ps_for_dma((uint16_t)(TOTAL_IMAGES - 1u));
        }

        status = run_parallel_split(n_ps, g_parallel_predictions, &run);
        if (status != XST_SUCCESS) {
            printf("DEPLOY run failed n_ps=%u status=%d\r\n", (unsigned)n_ps, status);
            return status;
        }

        status = svm_eval_accuracy_only(g_parallel_predictions,
                                        g_mnist_ground_truth,
                                        TOTAL_IMAGES,
                                        &final_acc,
                                        &final_mis);
        if (status != XST_SUCCESS) {
            printf("DEPLOY accuracy failed: %d\r\n", status);
            return status;
        }

        final_acc_x1e6 = (uint32_t)(final_acc * 1000000.0f + 0.5f);
        if (final_acc_x1e6 > 1000000u) {
            final_acc_x1e6 = 1000000u;
        }
        final_acc_ok = (final_acc >= TB_MIN_ACCURACY) ? 1 : 0;

        wall_us = cycles_to_us(run.total_cycles);
        ps_us = cycles_to_us(run.ps_cycles);
        pl_us = cycles_to_us(run.pl_cycles);
        gap_us = cycles_to_us(u64_abs_diff(run.ps_cycles, run.pl_cycles));
        makespan_us = (ps_us > pl_us) ? ps_us : pl_us;
        ps_share_x1000 = ((uint64_t)run.n_ps * 1000ull + (TOTAL_IMAGES / 2u)) / (uint64_t)TOTAL_IMAGES;

        if (makespan_us != 0u) {
            makespan_ips_x1000 = ((uint64_t)TOTAL_IMAGES * 1000000000ull + (makespan_us / 2ull)) / makespan_us;
        }

        printf("DEPLOY_CONFIG n_ps=%u n_pl=%u ps_share=%llu.%01llu%%\r\n",
               (unsigned)run.n_ps,
               (unsigned)run.n_pl,
               (unsigned long long)(ps_share_x1000 / 10ull),
               (unsigned long long)(ps_share_x1000 % 10ull));
        printf("DEPLOY_RESULT makespan_us=%llu wall_us=%llu ps_us=%llu pl_us=%llu gap_us=%llu images_per_s_by_max=%llu.%03llu\r\n",
               (unsigned long long)makespan_us,
               (unsigned long long)wall_us,
               (unsigned long long)ps_us,
               (unsigned long long)pl_us,
               (unsigned long long)gap_us,
               (unsigned long long)(makespan_ips_x1000 / 1000ull),
               (unsigned long long)(makespan_ips_x1000 % 1000ull));

#if DEPLOY_REPORT_SPLIT_SERIAL
        status = run_serial_split(run.n_ps,
                                  g_parallel_predictions,
                                  &serial_pl_cycles,
                                  &serial_ps_cycles,
                                  &serial_total_cycles);
        if (status != XST_SUCCESS) {
            printf("DEPLOY serial_ref failed n_ps=%u status=%d\r\n", (unsigned)run.n_ps, status);
            return status;
        }
        if (run.total_cycles != 0u) {
            speedup_vs_split_serial_x1000 =
                (serial_total_cycles * 1000ull + (run.total_cycles / 2ull)) / run.total_cycles;
        }
        printf("DEPLOY_REF split_serial_us=%llu split_serial_ps_us=%llu split_serial_pl_us=%llu speedup_vs_split_serial=%llu.%03llu\r\n",
               (unsigned long long)cycles_to_us(serial_total_cycles),
               (unsigned long long)cycles_to_us(serial_ps_cycles),
               (unsigned long long)cycles_to_us(serial_pl_cycles),
               (unsigned long long)(speedup_vs_split_serial_x1000 / 1000ull),
               (unsigned long long)(speedup_vs_split_serial_x1000 % 1000ull));
#endif

        printf("DEPLOY_ACC mismatches=%u accuracy=%u.%06u threshold=0.98 acc_ok=%d\r\n",
               (unsigned)final_mis,
               (unsigned)(final_acc_x1e6 / 1000000u),
               (unsigned)(final_acc_x1e6 % 1000000u),
               final_acc_ok);

        return final_acc_ok ? 0 : 1;
    }
#endif

    /* 1) Baseline throughput calibration on full 2601 images. */
    status = svm_run_batch_timed(g_mnist_test_q7_1, g_pl_predictions, TOTAL_IMAGES, &pl_dma_cycles, &pl_kernel_cycles);
    if (status != XST_SUCCESS) {
        printf("PL baseline failed: %d\r\n", status);
        return status;
    }
    pl_cycles = (pl_dma_cycles > pl_kernel_cycles) ? pl_dma_cycles : pl_kernel_cycles;

    status = svm_cpu_quantized_run_batch_timed(g_mnist_test_q7_1, g_cpu_predictions, TOTAL_IMAGES, &ps_cycles);
    if (status != XST_SUCCESS) {
        printf("PS baseline failed: %d\r\n", status);
        return status;
    }

    pl_cycles_per_img = (double)pl_cycles / (double)TOTAL_IMAGES;
    ps_cycles_per_img = (double)ps_cycles / (double)TOTAL_IMAGES;
    n_ps_equal_est = (uint16_t)(((double)TOTAL_IMAGES * pl_cycles_per_img) /
                                (pl_cycles_per_img + ps_cycles_per_img) + 0.5);
    n_ps_equal_est = align_n_ps_for_dma(n_ps_equal_est);

    printf("SPLIT_BASELINE pl_us=%llu ps_us=%llu est_n_ps=%u est_n_pl=%u\r\n",
           (unsigned long long)cycles_to_us(pl_cycles),
           (unsigned long long)cycles_to_us(ps_cycles),
           (unsigned)n_ps_equal_est,
           (unsigned)(TOTAL_IMAGES - n_ps_equal_est));

    /* 2) Explore nearby splits. */
    candidate_count = build_split_candidates(n_ps_equal_est, candidates, SPLIT_SCAN_MAX_CANDIDATES);
    if (candidate_count == 0u) {
        printf("No split candidates\r\n");
        return XST_FAILURE;
    }

    for (uint32_t i = 0u; i < candidate_count; ++i) {
        split_run_result_t cur;
        uint64_t cur_gap;

        status = run_parallel_split(candidates[i], g_parallel_predictions, &cur);
        if (status != XST_SUCCESS) {
            printf("SPLIT_SCAN skip n_ps=%u status=%d\r\n", (unsigned)candidates[i], status);
            continue;
        }

        cur_gap = u64_abs_diff(cur.ps_cycles, cur.pl_cycles);
        printf("SPLIT_SCAN n_ps=%u n_pl=%u total_us=%llu ps_us=%llu pl_us=%llu gap_us=%llu\r\n",
               (unsigned)cur.n_ps,
               (unsigned)cur.n_pl,
               (unsigned long long)cycles_to_us(cur.total_cycles),
               (unsigned long long)cycles_to_us(cur.ps_cycles),
               (unsigned long long)cycles_to_us(cur.pl_cycles),
               (unsigned long long)cycles_to_us(cur_gap));

        valid_runs[valid_count++] = cur;
        if (valid_count >= SPLIT_SCAN_MAX_CANDIDATES) {
            break;
        }
    }

    if (valid_count == 0u) {
        printf("SPLIT_SCAN no valid candidate\r\n");
        return XST_FAILURE;
    }

    /*
     * Selection policy:
     * - First find minimal total latency.
     * - Then, within a small window of that fastest point, choose the smallest
     *   PS/PL finish-time gap for better overlap balance.
     */
    {
        uint64_t fastest_total = valid_runs[0].total_cycles;
        uint64_t total_limit;

        for (uint32_t i = 1u; i < valid_count; ++i) {
            if (valid_runs[i].total_cycles < fastest_total) {
                fastest_total = valid_runs[i].total_cycles;
            }
        }

        total_limit = fastest_total + (fastest_total * SPLIT_BALANCE_WINDOW_PCT) / 100u;

        for (uint32_t i = 0u; i < valid_count; ++i) {
            split_run_result_t cur = valid_runs[i];
            uint64_t cur_gap = u64_abs_diff(cur.ps_cycles, cur.pl_cycles);
            uint64_t best_gap;

            if (cur.total_cycles > total_limit) {
                continue;
            }
            if (!best_valid) {
                best_run = cur;
                best_valid = 1;
                continue;
            }

            best_gap = u64_abs_diff(best_run.ps_cycles, best_run.pl_cycles);
            if ((cur_gap < best_gap) ||
                ((cur_gap == best_gap) && (cur.total_cycles < best_run.total_cycles))) {
                best_run = cur;
            }
        }

        if (!best_valid) {
            /* Fallback: strict fastest if all candidates were outside window. */
            best_run = valid_runs[0];
            for (uint32_t i = 1u; i < valid_count; ++i) {
                if (valid_runs[i].total_cycles < best_run.total_cycles) {
                    best_run = valid_runs[i];
                }
            }
            best_valid = 1;
        }

        printf("SPLIT_SELECT valid=%u fastest_us=%llu window=%u%%\r\n",
               (unsigned)valid_count,
               (unsigned long long)cycles_to_us(fastest_total),
               (unsigned)SPLIT_BALANCE_WINDOW_PCT);
    }

    /* 3) Final rerun on selected split for final report/accuracy. */
    status = run_parallel_split(best_run.n_ps, g_parallel_predictions, &best_run);
    if (status != XST_SUCCESS) {
        printf("FINAL split run failed n_ps=%u status=%d\r\n", (unsigned)best_run.n_ps, status);
        return status;
    }

    status = svm_eval_accuracy_only(g_parallel_predictions,
                                    g_mnist_ground_truth,
                                    TOTAL_IMAGES,
                                    &final_acc,
                                    &final_mis);
    if (status != XST_SUCCESS) {
        printf("FINAL accuracy failed: %d\r\n", status);
        return status;
    }

    final_acc_x1e6 = (uint32_t)(final_acc * 1000000.0f + 0.5f);
    if (final_acc_x1e6 > 1000000u) {
        final_acc_x1e6 = 1000000u;
    }
    final_acc_ok = (final_acc >= TB_MIN_ACCURACY) ? 1 : 0;

    {
        uint64_t total_us = cycles_to_us(best_run.total_cycles);
        uint64_t ps_us = cycles_to_us(best_run.ps_cycles);
        uint64_t pl_us = cycles_to_us(best_run.pl_cycles);
        uint64_t gap_us = cycles_to_us(u64_abs_diff(best_run.ps_cycles, best_run.pl_cycles));
        uint64_t total_ips_x1000 = 0u;
        uint64_t speedup_vs_split_serial_x1000 = 0u;
        uint64_t serial_total_cycles = 0u;
        uint64_t serial_ps_cycles = 0u;
        uint64_t serial_pl_cycles = 0u;
        uint64_t serial_total_us;
        uint64_t ps_share_x1000 = ((uint64_t)best_run.n_ps * 1000ull + (TOTAL_IMAGES / 2u)) / (uint64_t)TOTAL_IMAGES;

        status = run_serial_split(best_run.n_ps,
                                  g_parallel_predictions,
                                  &serial_pl_cycles,
                                  &serial_ps_cycles,
                                  &serial_total_cycles);
        if (status != XST_SUCCESS) {
            printf("SERIAL_SPLIT failed n_ps=%u status=%d\r\n", (unsigned)best_run.n_ps, status);
            return status;
        }

        if (best_run.total_cycles != 0u) {
            total_ips_x1000 = ((uint64_t)TOTAL_IMAGES * (uint64_t)COUNTS_PER_SECOND * 1000ull +
                               (best_run.total_cycles / 2ull)) / best_run.total_cycles;
            speedup_vs_split_serial_x1000 =
                (serial_total_cycles * 1000ull + (best_run.total_cycles / 2ull)) / best_run.total_cycles;
        }
        serial_total_us = cycles_to_us(serial_total_cycles);

        printf("PAR_FINAL n_ps=%u n_pl=%u ps_share=%llu.%01llu%%\r\n",
               (unsigned)best_run.n_ps,
               (unsigned)best_run.n_pl,
               (unsigned long long)(ps_share_x1000 / 10ull),
               (unsigned long long)(ps_share_x1000 % 10ull));
        printf("PAR_FINAL total_us=%llu ps_us=%llu pl_us=%llu gap_us=%llu\r\n",
               (unsigned long long)total_us,
               (unsigned long long)ps_us,
               (unsigned long long)pl_us,
               (unsigned long long)gap_us);
        printf("PAR_FINAL split_serial_us=%llu split_serial_ps_us=%llu split_serial_pl_us=%llu\r\n",
               (unsigned long long)serial_total_us,
               (unsigned long long)cycles_to_us(serial_ps_cycles),
               (unsigned long long)cycles_to_us(serial_pl_cycles));
        printf("PAR_FINAL images_per_s=%llu.%03llu speedup_vs_split_serial=%llu.%03llu\r\n",
               (unsigned long long)(total_ips_x1000 / 1000ull),
               (unsigned long long)(total_ips_x1000 % 1000ull),
               (unsigned long long)(speedup_vs_split_serial_x1000 / 1000ull),
               (unsigned long long)(speedup_vs_split_serial_x1000 % 1000ull));
        printf("PAR_FINAL mismatches=%u accuracy=%u.%06u threshold=0.98 acc_ok=%d\r\n",
               (unsigned)final_mis,
               (unsigned)(final_acc_x1e6 / 1000000u),
               (unsigned)(final_acc_x1e6 % 1000000u),
               final_acc_ok);
    }

    return final_acc_ok ? 0 : 1;
}
