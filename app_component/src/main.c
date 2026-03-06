#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mnist_q7_1_data.h"
#include "svm_cpu_quantized.h"
#include "svm_cpu_model_data.h"
#include "svm_ps_driver.h"
#include "xil_cache.h"
#include "xil_cache_l.h"
#include "xl2cc_counter.h"
#include "xpm_counter.h"
#include "xparameters.h"
#include "xstatus.h"

#ifndef XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#ifndef XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#include "xtime_l.h"

/* Match TB acceptance rule: only print/pass by accuracy threshold. */
#define TB_MIN_ACCURACY 0.98f

/* PMU profiling switches (cfg11 mandatory pass, cfg5/cfg7 optional extra passes). */
#define ENABLE_PMU_PROFILING 1u
#define ENABLE_PMU_MULTI_CONFIG 1u

/*
 * Concurrency proof mode:
 * Runs PL_ONLY / PS_ONLY / PARALLEL sweeps repeatedly and reports medians.
 * Uses high-frequency async polling hook while PS is running.
 */
#define ENABLE_CONCURRENCY_PROOF_TEST 1u
#define CONCURRENCY_PROOF_RUNS 30u
#define CONCURRENCY_PROOF_WARMUP 3u
#define CONCURRENCY_PROOF_HOOK_PERIOD_IMAGES 1u
#define CONCURRENCY_PROOF_PS_IMAGES ((uint16_t)SVM_CPU_NUM_IMAGES)
#define CONCURRENCY_PROOF_MEM_STRIDE_BYTES 64u

static uint8_t g_pl_predictions[MNIST_NUM_IMAGES] __attribute__((aligned(64)));
static uint8_t g_cpu_predictions[SVM_CPU_NUM_IMAGES] __attribute__((aligned(64)));

#if ENABLE_PMU_PROFILING
/*
 * Run one timed PS profiling pass under a selected PMU event config.
 * This wraps the same compute function so we can compare event sets
 * without changing kernel behavior.
 */
static int run_cpu_profile_pass(s32 pmu_cfg,
                                int enable_l2_counter,
                                s32 l2_event0,
                                s32 l2_event1,
                                const int8_t *in_q7_1,
                                uint8_t *out_label,
                                uint16_t n_images,
                                uint64_t *cycles_out,
                                u32 pmu_evt_out[6],
                                u32 *l2_evt0_out,
                                u32 *l2_evt1_out) {
    int status;

    if ((cycles_out == NULL) || (pmu_evt_out == NULL)) {
        return XST_INVALID_PARAM;
    }

    memset(pmu_evt_out, 0, sizeof(u32) * 6u);
    if (l2_evt0_out != NULL) {
        *l2_evt0_out = 0u;
    }
    if (l2_evt1_out != NULL) {
        *l2_evt1_out = 0u;
    }

    Xpm_SetEvents(pmu_cfg);
    if (enable_l2_counter != 0) {
        XL2cc_EventCtrInit(l2_event0, l2_event1);
        XL2cc_EventCtrStart();
    }

    status = svm_cpu_quantized_run_batch_timed(in_q7_1, out_label, n_images, cycles_out);

    Xpm_GetEventCounters(pmu_evt_out);
    if (enable_l2_counter != 0) {
        u32 c0 = 0u;
        u32 c1 = 0u;
        XL2cc_EventCtrStop(&c0, &c1);
        if (l2_evt0_out != NULL) {
            *l2_evt0_out = c0;
        }
        if (l2_evt1_out != NULL) {
            *l2_evt1_out = c1;
        }
    }
    Xpm_DisableEventCounters();

    return status;
}
#endif

#if ENABLE_CONCURRENCY_PROOF_TEST
typedef struct {
    uint64_t pl_done_cycles;
    int pl_done_valid;
} proof_poll_ctx_t;

static inline uint64_t proof_max_u64(uint64_t a, uint64_t b) {
    return (a > b) ? a : b;
}

static inline uint64_t proof_cycles_to_us(uint64_t cycles) {
    return (cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
}

static void proof_sort_u64(uint64_t *arr, uint32_t n) {
    for (uint32_t i = 1u; i < n; ++i) {
        uint64_t key = arr[i];
        uint32_t j = i;
        while ((j > 0u) && (arr[j - 1u] > key)) {
            arr[j] = arr[j - 1u];
            --j;
        }
        arr[j] = key;
    }
}

static uint64_t proof_median_u64(const uint64_t *arr, uint32_t n) {
    uint64_t tmp[CONCURRENCY_PROOF_RUNS];
    if ((arr == NULL) || (n == 0u) || (n > CONCURRENCY_PROOF_RUNS)) {
        return 0u;
    }
    for (uint32_t i = 0u; i < n; ++i) {
        tmp[i] = arr[i];
    }
    proof_sort_u64(tmp, n);
    if ((n & 1u) != 0u) {
        return tmp[n / 2u];
    }
    return (tmp[(n / 2u) - 1u] + tmp[n / 2u]) / 2u;
}

static void proof_pl_poll_hook(uint32_t img_idx, void *user) {
    proof_poll_ctx_t *ctx = (proof_poll_ctx_t *)user;
    svm_pl_async_status_t st;
    (void)img_idx;

    if ((ctx == NULL) || (ctx->pl_done_valid != 0)) {
        return;
    }
    if (svm_run_batch_async_poll(&st) != XST_SUCCESS) {
        return;
    }
    if ((st.s2mm_done != 0u) && (st.ip_done != 0u)) {
        ctx->pl_done_cycles = proof_max_u64(st.dma_cycles, st.kernel_cycles);
        ctx->pl_done_valid = 1;
    }
}

static volatile uint32_t g_proof_sink = 0u;

static void proof_cpu_spin_for_cycles(uint64_t budget_cycles, proof_poll_ctx_t *poll_ctx) {
    XTime t0;
    XTime tnow;
    uint32_t x = 0x13579BDFu;

    if (budget_cycles == 0u) {
        return;
    }

    XTime_GetTime(&t0);
    tnow = t0;
    while ((uint64_t)(tnow - t0) < budget_cycles) {
        if (poll_ctx != NULL) {
            proof_pl_poll_hook(0u, poll_ctx);
        }
        for (uint32_t i = 0u; i < 512u; ++i) {
            x = (x * 1664525u) + 1013904223u;
            x ^= (x >> 11);
            x += 0x9E3779B9u;
        }
        XTime_GetTime(&tnow);
    }
    if (poll_ctx != NULL) {
        proof_pl_poll_hook(0u, poll_ctx);
    }

    g_proof_sink ^= x;
}

static void proof_cpu_mem_stress_for_cycles(const int8_t *buf,
                                            uint32_t len_bytes,
                                            uint64_t budget_cycles,
                                            proof_poll_ctx_t *poll_ctx) {
    XTime t0;
    XTime tnow;
    volatile uint32_t acc = 0u;

    if ((buf == NULL) || (len_bytes < CONCURRENCY_PROOF_MEM_STRIDE_BYTES) || (budget_cycles == 0u)) {
        return;
    }

    XTime_GetTime(&t0);
    tnow = t0;
    while ((uint64_t)(tnow - t0) < budget_cycles) {
        if (poll_ctx != NULL) {
            proof_pl_poll_hook(0u, poll_ctx);
        }
        for (uint32_t i = 0u; i < len_bytes; i += CONCURRENCY_PROOF_MEM_STRIDE_BYTES) {
            acc += (uint8_t)buf[i];
        }
        XTime_GetTime(&tnow);
    }
    if (poll_ctx != NULL) {
        proof_pl_poll_hook(0u, poll_ctx);
    }

    g_proof_sink ^= (uint32_t)acc;
}

static int run_concurrency_proof_test(void) {
    const uint16_t n_pl = (uint16_t)MNIST_NUM_IMAGES;
    const uint16_t n_ps = CONCURRENCY_PROOF_PS_IMAGES;
    uint64_t pl_only_us[CONCURRENCY_PROOF_RUNS];
    uint64_t ps_only_us[CONCURRENCY_PROOF_RUNS];
    uint64_t par_spin_pl_us[CONCURRENCY_PROOF_RUNS];
    uint64_t par_mem_pl_us[CONCURRENCY_PROOF_RUNS];
    uint64_t par_total_us[CONCURRENCY_PROOF_RUNS];
    uint64_t par_pl_us[CONCURRENCY_PROOF_RUNS];
    uint64_t par_ps_us[CONCURRENCY_PROOF_RUNS];
    uint32_t keep_idx = 0u;
    int status;

    if ((n_pl == 0u) || (n_pl > (uint16_t)MNIST_NUM_IMAGES) ||
        (n_ps == 0u) || (n_ps > (uint16_t)SVM_CPU_NUM_IMAGES)) {
        return XST_INVALID_PARAM;
    }

    printf("CONCURRENCY_PROOF runs=%u warmup=%u hook_period=%u n_pl=%u n_ps=%u\r\n",
           (unsigned)CONCURRENCY_PROOF_RUNS,
           (unsigned)CONCURRENCY_PROOF_WARMUP,
           (unsigned)CONCURRENCY_PROOF_HOOK_PERIOD_IMAGES,
           (unsigned)n_pl,
           (unsigned)n_ps);

    for (uint32_t run = 0u; run < CONCURRENCY_PROOF_RUNS; ++run) {
        uint64_t pl_dma_cycles = 0u;
        uint64_t pl_kernel_cycles = 0u;
        uint64_t pl_only_cycles;
        uint64_t ps_only_cycles = 0u;
        uint64_t par_spin_dma_cycles = 0u;
        uint64_t par_spin_kernel_cycles = 0u;
        uint64_t par_mem_dma_cycles = 0u;
        uint64_t par_mem_kernel_cycles = 0u;
        uint64_t par_dma_cycles = 0u;
        uint64_t par_kernel_cycles = 0u;
        uint64_t par_ps_cycles = 0u;
        uint64_t par_total_cycles = 0u;
        uint64_t par_spin_pl_cycles = 0u;
        uint64_t par_mem_pl_cycles = 0u;
        uint64_t par_pl_cycles = 0u;
        XTime t_par_start;
        XTime t_par_end;
        proof_poll_ctx_t poll_ctx_spin;
        proof_poll_ctx_t poll_ctx_mem;
        proof_poll_ctx_t poll_ctx_par;

        /* Case A: PL only baseline. */
        status = svm_run_batch_timed(g_mnist_test_q7_1, g_pl_predictions, n_pl, &pl_dma_cycles, &pl_kernel_cycles);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PL_ONLY status=%d\r\n", (unsigned)run, status);
            return status;
        }
        pl_only_cycles = proof_max_u64(pl_dma_cycles, pl_kernel_cycles);

        /* Case B: PS-only baseline. */
        svm_cpu_quantized_set_progress_hook(NULL, NULL, 1u);
        status = svm_cpu_quantized_run_batch_timed(g_mnist_test_q7_1, g_cpu_predictions, n_ps, &ps_only_cycles);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PS_ONLY status=%d\r\n", (unsigned)run, status);
            return status;
        }

        /*
         * Case C: PL + CPU compute-only spin for the same wall-time budget as PS_ONLY.
         * If this barely slows PL, CPU occupancy alone is not the root cause.
         */
        poll_ctx_spin.pl_done_cycles = 0u;
        poll_ctx_spin.pl_done_valid = 0;
        status = svm_run_batch_async_start(g_mnist_test_q7_1, g_pl_predictions, n_pl);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PAR_SPIN_START status=%d\r\n", (unsigned)run, status);
            return status;
        }
        proof_cpu_spin_for_cycles(ps_only_cycles, &poll_ctx_spin);
        status = svm_run_batch_async_wait(g_pl_predictions, n_pl, &par_spin_dma_cycles, &par_spin_kernel_cycles);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PAR_SPIN_WAIT status=%d\r\n", (unsigned)run, status);
            return status;
        }
        par_spin_pl_cycles = (poll_ctx_spin.pl_done_valid != 0)
                                 ? poll_ctx_spin.pl_done_cycles
                                 : proof_max_u64(par_spin_dma_cycles, par_spin_kernel_cycles);

        /*
         * Case D: PL + CPU memory-stream stress with the same wall-time budget.
         * This is a direct memory-arbitration control.
         */
        poll_ctx_mem.pl_done_cycles = 0u;
        poll_ctx_mem.pl_done_valid = 0;
        status = svm_run_batch_async_start(g_mnist_test_q7_1, g_pl_predictions, n_pl);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PAR_MEM_START status=%d\r\n", (unsigned)run, status);
            return status;
        }
        proof_cpu_mem_stress_for_cycles(g_mnist_test_q7_1,
                                        (uint32_t)MNIST_INPUT_SIZE_BYTES,
                                        ps_only_cycles,
                                        &poll_ctx_mem);
        status = svm_run_batch_async_wait(g_pl_predictions, n_pl, &par_mem_dma_cycles, &par_mem_kernel_cycles);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PAR_MEM_WAIT status=%d\r\n", (unsigned)run, status);
            return status;
        }
        par_mem_pl_cycles = (poll_ctx_mem.pl_done_valid != 0)
                                ? poll_ctx_mem.pl_done_cycles
                                : proof_max_u64(par_mem_dma_cycles, par_mem_kernel_cycles);

        /* Case E: PL + real PS SVM inference + high-frequency PL polling. */
        poll_ctx_par.pl_done_cycles = 0u;
        poll_ctx_par.pl_done_valid = 0;
        XTime_GetTime(&t_par_start);
        status = svm_run_batch_async_start(g_mnist_test_q7_1, g_pl_predictions, n_pl);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PAR_START status=%d\r\n", (unsigned)run, status);
            return status;
        }

        svm_cpu_quantized_set_progress_hook(proof_pl_poll_hook, &poll_ctx_par, CONCURRENCY_PROOF_HOOK_PERIOD_IMAGES);
        status = svm_cpu_quantized_run_batch_timed(g_mnist_test_q7_1, g_cpu_predictions, n_ps, &par_ps_cycles);
        svm_cpu_quantized_set_progress_hook(NULL, NULL, 1u);
        if (status != XST_SUCCESS) {
            uint64_t drain_dma = 0u;
            uint64_t drain_kernel = 0u;
            (void)svm_run_batch_async_wait(g_pl_predictions, n_pl, &drain_dma, &drain_kernel);
            printf("PROOF FAIL run=%u phase=PAR_PS status=%d\r\n", (unsigned)run, status);
            return status;
        }

        status = svm_run_batch_async_wait(g_pl_predictions, n_pl, &par_dma_cycles, &par_kernel_cycles);
        if (status != XST_SUCCESS) {
            printf("PROOF FAIL run=%u phase=PAR_WAIT status=%d\r\n", (unsigned)run, status);
            return status;
        }
        XTime_GetTime(&t_par_end);

        par_total_cycles = (uint64_t)(t_par_end - t_par_start);
        par_pl_cycles = (poll_ctx_par.pl_done_valid != 0)
                            ? poll_ctx_par.pl_done_cycles
                            : proof_max_u64(par_dma_cycles, par_kernel_cycles);

        if (run >= CONCURRENCY_PROOF_WARMUP) {
            pl_only_us[keep_idx] = proof_cycles_to_us(pl_only_cycles);
            ps_only_us[keep_idx] = proof_cycles_to_us(ps_only_cycles);
            par_spin_pl_us[keep_idx] = proof_cycles_to_us(par_spin_pl_cycles);
            par_mem_pl_us[keep_idx] = proof_cycles_to_us(par_mem_pl_cycles);
            par_total_us[keep_idx] = proof_cycles_to_us(par_total_cycles);
            par_pl_us[keep_idx] = proof_cycles_to_us(par_pl_cycles);
            par_ps_us[keep_idx] = proof_cycles_to_us(par_ps_cycles);
            ++keep_idx;
        }

        printf("PROOF run=%u pl_only_us=%llu ps_only_us=%llu spin_pl_us=%llu mem_pl_us=%llu par_total_us=%llu par_pl_us=%llu par_ps_us=%llu ps_earlier=%u latch=%u\r\n",
               (unsigned)run,
               (unsigned long long)proof_cycles_to_us(pl_only_cycles),
               (unsigned long long)proof_cycles_to_us(ps_only_cycles),
               (unsigned long long)proof_cycles_to_us(par_spin_pl_cycles),
               (unsigned long long)proof_cycles_to_us(par_mem_pl_cycles),
               (unsigned long long)proof_cycles_to_us(par_total_cycles),
               (unsigned long long)proof_cycles_to_us(par_pl_cycles),
               (unsigned long long)proof_cycles_to_us(par_ps_cycles),
               (unsigned)((par_ps_cycles < par_pl_cycles) ? 1u : 0u),
               (unsigned)((poll_ctx_par.pl_done_valid != 0) ? 1u : 0u));
    }

    if (keep_idx == 0u) {
        return XST_FAILURE;
    }

    {
        const uint64_t med_pl_only_us = proof_median_u64(pl_only_us, keep_idx);
        const uint64_t med_ps_only_us = proof_median_u64(ps_only_us, keep_idx);
        const uint64_t med_spin_pl_us = proof_median_u64(par_spin_pl_us, keep_idx);
        const uint64_t med_mem_pl_us = proof_median_u64(par_mem_pl_us, keep_idx);
        const uint64_t med_par_total_us = proof_median_u64(par_total_us, keep_idx);
        const uint64_t med_par_pl_us = proof_median_u64(par_pl_us, keep_idx);
        const uint64_t med_par_ps_us = proof_median_u64(par_ps_us, keep_idx);
        const uint64_t spin_slowdown_x1000 = (med_pl_only_us != 0u) ? ((med_spin_pl_us * 1000ull + (med_pl_only_us / 2ull)) / med_pl_only_us) : 0u;
        const uint64_t mem_slowdown_x1000 = (med_pl_only_us != 0u) ? ((med_mem_pl_us * 1000ull + (med_pl_only_us / 2ull)) / med_pl_only_us) : 0u;
        const uint64_t pl_slowdown_x1000 = (med_pl_only_us != 0u) ? ((med_par_pl_us * 1000ull + (med_pl_only_us / 2ull)) / med_pl_only_us) : 0u;
        const uint64_t ps_slowdown_x1000 = (med_ps_only_us != 0u) ? ((med_par_ps_us * 1000ull + (med_ps_only_us / 2ull)) / med_ps_only_us) : 0u;
        const uint64_t serial_sum_us = med_pl_only_us + med_ps_only_us;
        const uint64_t parallel_gain_x1000 = (med_par_total_us != 0u) ? ((serial_sum_us * 1000ull + (med_par_total_us / 2ull)) / med_par_total_us) : 0u;
        const int mem_arbiter_claim =
            ((mem_slowdown_x1000 > (spin_slowdown_x1000 + 100ull)) &&
             (pl_slowdown_x1000 > (spin_slowdown_x1000 + 100ull)))
                ? 1
                : 0;
        float pl_acc = 0.0f;
        float ps_acc = 0.0f;
        uint32_t pl_mis = 0u;
        uint32_t ps_mis = 0u;

        (void)svm_eval_accuracy_only(g_pl_predictions, g_mnist_ground_truth, n_pl, &pl_acc, &pl_mis);
        (void)svm_cpu_quantized_eval_accuracy(g_cpu_predictions, g_svm_cpu_ground_truth, n_ps, &ps_acc, &ps_mis);

        printf("PROOF MEDIAN pl_only_us=%llu ps_only_us=%llu spin_pl_us=%llu mem_pl_us=%llu par_total_us=%llu par_pl_us=%llu par_ps_us=%llu\r\n",
               (unsigned long long)med_pl_only_us,
               (unsigned long long)med_ps_only_us,
               (unsigned long long)med_spin_pl_us,
               (unsigned long long)med_mem_pl_us,
               (unsigned long long)med_par_total_us,
               (unsigned long long)med_par_pl_us,
               (unsigned long long)med_par_ps_us);
        printf("PROOF RATIOS spin_slowdown=%llu.%03llu mem_slowdown=%llu.%03llu pl_slowdown=%llu.%03llu ps_slowdown=%llu.%03llu serial_over_parallel=%llu.%03llu\r\n",
               (unsigned long long)(spin_slowdown_x1000 / 1000ull),
               (unsigned long long)(spin_slowdown_x1000 % 1000ull),
               (unsigned long long)(mem_slowdown_x1000 / 1000ull),
               (unsigned long long)(mem_slowdown_x1000 % 1000ull),
               (unsigned long long)(pl_slowdown_x1000 / 1000ull),
               (unsigned long long)(pl_slowdown_x1000 % 1000ull),
               (unsigned long long)(ps_slowdown_x1000 / 1000ull),
               (unsigned long long)(ps_slowdown_x1000 % 1000ull),
               (unsigned long long)(parallel_gain_x1000 / 1000ull),
               (unsigned long long)(parallel_gain_x1000 % 1000ull));
        printf("PROOF CLAIM memory_arbiter=%d sink=%u\r\n",
               mem_arbiter_claim,
               (unsigned)g_proof_sink);
        printf("PROOF ACC n_pl=%u n_ps=%u pl_mis=%u pl_acc=%u.%06u ps_mis=%u ps_acc=%u.%06u\r\n",
               (unsigned)n_pl,
               (unsigned)n_ps,
               (unsigned)pl_mis,
               (unsigned)((uint32_t)(pl_acc * 1000000.0f + 0.5f) / 1000000u),
               (unsigned)((uint32_t)(pl_acc * 1000000.0f + 0.5f) % 1000000u),
               (unsigned)ps_mis,
               (unsigned)((uint32_t)(ps_acc * 1000000.0f + 0.5f) / 1000000u),
               (unsigned)((uint32_t)(ps_acc * 1000000.0f + 0.5f) % 1000000u));
    }

    return XST_SUCCESS;
}
#endif

int main(void) {
    int status;
    uint64_t pl_dma_cycles = 0u;
    uint64_t pl_kernel_cycles = 0u;
    uint64_t pl_dma_time_us = 0u;
    uint64_t pl_kernel_time_us = 0u;
    uint64_t pl_images_per_s_x1000 = 0u;
    float pl_accuracy = 0.0f;
    uint32_t pl_mismatches = 0u;
    uint32_t pl_accuracy_x1e6 = 0u;
    int pl_acc_ok = 0;

    uint64_t cpu_cycles = 0u;
    uint64_t cpu_time_us = 0u;
    uint64_t cpu_images_per_s_x1000 = 0u;
    float cpu_accuracy = 0.0f;
    uint32_t cpu_mismatches = 0u;
    uint32_t cpu_accuracy_x1e6 = 0u;
    int cpu_acc_ok = 0;
#if ENABLE_PMU_PROFILING
    u32 pmu_evt11[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    u32 l2_drreq = 0u;
    u32 l2_drhit = 0u;
    uint64_t l1d_refill_per_kaccess_x1000 = 0u;
    uint64_t l2_read_hit_rate_x1000 = 0u;
    uint64_t data_stall_per_img_x1000 = 0u;
    uint64_t instr_refill_per_img_x1000 = 0u;
#if ENABLE_PMU_MULTI_CONFIG
    u32 pmu_evt5[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    u32 pmu_evt7[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint64_t cpu_cycles_cfg5 = 0u;
    uint64_t cpu_cycles_cfg7 = 0u;
    uint64_t nodispatch_ratio_x1000 = 0u;
    uint64_t issueempty_ratio_x1000 = 0u;
    uint64_t pldstall_ratio_x1000 = 0u;
    uint64_t writestall_ratio_x1000 = 0u;
    uint64_t neonrename_per_img_x1000 = 0u;
#endif
#endif

    printf("Starting Program\r\n");
    Xil_ICacheEnable();
    Xil_DCacheEnable();
    Xil_L2CacheEnable();

    /* Phase 1: initialize HW blocks used by PL path (AXI DMA + SVM IP). */
    status = svm_init_hw();
    if (status != XST_SUCCESS) {
        printf("svm_init_hw failed: %d\r\n", status);
        return status;
    }

#if ENABLE_CONCURRENCY_PROOF_TEST
    status = run_concurrency_proof_test();
    if (status != XST_SUCCESS) {
        printf("run_concurrency_proof_test failed: %d\r\n", status);
        return status;
    }
    return 0;
#endif

    /* Phase 2: run PL batch first (serial mode baseline). */
    status = svm_run_batch_timed(g_mnist_test_q7_1,
                                 g_pl_predictions,
                                 (uint16_t)MNIST_NUM_IMAGES,
                                 &pl_dma_cycles,
                                 &pl_kernel_cycles);
    if (status != XST_SUCCESS) {
        printf("svm_run_batch_timed failed at N=%u: %d\r\n", (unsigned)MNIST_NUM_IMAGES, status);
        return status;
    }

    /* Convert cycle counters to microseconds and throughput. */
    pl_kernel_time_us = (pl_kernel_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
    pl_dma_time_us = (pl_dma_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
    if (pl_dma_cycles != 0u) {
        pl_images_per_s_x1000 = ((uint64_t)MNIST_NUM_IMAGES * (uint64_t)COUNTS_PER_SECOND * 1000ull +
                                 (pl_dma_cycles / 2ull)) / pl_dma_cycles;
    }

    /* PL accuracy check against MNIST ground truth (bit0 comparison). */
    status = svm_eval_accuracy_only(g_pl_predictions,
                                    g_mnist_ground_truth,
                                    (uint16_t)MNIST_NUM_IMAGES,
                                    &pl_accuracy,
                                    &pl_mismatches);
    if (status != XST_SUCCESS) {
        printf("svm_eval_accuracy_only failed: %d\r\n", status);
        return status;
    }

    pl_accuracy_x1e6 = (uint32_t)(pl_accuracy * 1000000.0f + 0.5f);
    if (pl_accuracy_x1e6 > 1000000u) {
        pl_accuracy_x1e6 = 1000000u;
    }
    pl_acc_ok = (pl_accuracy >= TB_MIN_ACCURACY) ? 1 : 0;

    /* Phase 3: run PS quantized kernel (with optional PMU profiling passes). */
#if ENABLE_PMU_PROFILING
    status = run_cpu_profile_pass(XPM_CNTRCFG11,
                                  1,
                                  XL2CC_DRREQ,
                                  XL2CC_DRHIT,
                                  g_mnist_test_q7_1,
                                  g_cpu_predictions,
                                  (uint16_t)SVM_CPU_NUM_IMAGES,
                                  &cpu_cycles,
                                  pmu_evt11,
                                  &l2_drreq,
                                  &l2_drhit);
#if ENABLE_PMU_MULTI_CONFIG
    if (status == XST_SUCCESS) {
        status = run_cpu_profile_pass(XPM_CNTRCFG5,
                                      0,
                                      0,
                                      0,
                                      g_mnist_test_q7_1,
                                      g_cpu_predictions,
                                      (uint16_t)SVM_CPU_NUM_IMAGES,
                                      &cpu_cycles_cfg5,
                                      pmu_evt5,
                                      NULL,
                                      NULL);
    }
    if (status == XST_SUCCESS) {
        status = run_cpu_profile_pass(XPM_CNTRCFG7,
                                      0,
                                      0,
                                      0,
                                      g_mnist_test_q7_1,
                                      g_cpu_predictions,
                                      (uint16_t)SVM_CPU_NUM_IMAGES,
                                      &cpu_cycles_cfg7,
                                      pmu_evt7,
                                      NULL,
                                      NULL);
    }
#endif
#else
    status = svm_cpu_quantized_run_batch_timed(g_mnist_test_q7_1,
                                               g_cpu_predictions,
                                               (uint16_t)SVM_CPU_NUM_IMAGES,
                                               &cpu_cycles);
#endif
    if (status != XST_SUCCESS) {
        printf("cpu_run failed at N=%u: %d\r\n", (unsigned)SVM_CPU_NUM_IMAGES, status);
        return status;
    }

    /* Phase 4: PS timing and PMU derived metrics formatting. */
    cpu_time_us = (cpu_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
    if (cpu_cycles != 0u) {
        cpu_images_per_s_x1000 = ((uint64_t)SVM_CPU_NUM_IMAGES * (uint64_t)COUNTS_PER_SECOND * 1000ull +
                                  (cpu_cycles / 2ull)) / cpu_cycles;
    }
#if ENABLE_PMU_PROFILING
    if (pmu_evt11[4] != 0u) {
        l1d_refill_per_kaccess_x1000 =
            ((uint64_t)pmu_evt11[3] * 1000000ull + ((uint64_t)pmu_evt11[4] / 2ull)) /
            (uint64_t)pmu_evt11[4];
    }
    if (l2_drreq != 0u) {
        l2_read_hit_rate_x1000 =
            ((uint64_t)l2_drhit * 1000ull + ((uint64_t)l2_drreq / 2ull)) / (uint64_t)l2_drreq;
    }
    data_stall_per_img_x1000 =
        ((uint64_t)pmu_evt11[0] * 1000ull + ((uint64_t)SVM_CPU_NUM_IMAGES / 2ull)) /
        (uint64_t)SVM_CPU_NUM_IMAGES;
    instr_refill_per_img_x1000 =
        ((uint64_t)pmu_evt11[1] * 1000ull + ((uint64_t)SVM_CPU_NUM_IMAGES / 2ull)) /
        (uint64_t)SVM_CPU_NUM_IMAGES;
#if ENABLE_PMU_MULTI_CONFIG
    if (cpu_cycles_cfg5 != 0u) {
        nodispatch_ratio_x1000 =
            ((uint64_t)pmu_evt5[4] * 1000ull + (cpu_cycles_cfg5 / 2ull)) / cpu_cycles_cfg5;
        issueempty_ratio_x1000 =
            ((uint64_t)pmu_evt5[5] * 1000ull + (cpu_cycles_cfg5 / 2ull)) / cpu_cycles_cfg5;
    }
    if (cpu_cycles_cfg7 != 0u) {
        pldstall_ratio_x1000 =
            ((uint64_t)pmu_evt7[1] * 1000ull + (cpu_cycles_cfg7 / 2ull)) / cpu_cycles_cfg7;
        writestall_ratio_x1000 =
            ((uint64_t)pmu_evt7[2] * 1000ull + (cpu_cycles_cfg7 / 2ull)) / cpu_cycles_cfg7;
    }
    neonrename_per_img_x1000 =
        ((uint64_t)pmu_evt7[0] * 1000ull + ((uint64_t)SVM_CPU_NUM_IMAGES / 2ull)) /
        (uint64_t)SVM_CPU_NUM_IMAGES;
#endif
#endif

    /* PS accuracy check against model-aligned label table. */
    status = svm_cpu_quantized_eval_accuracy(g_cpu_predictions,
                                             g_svm_cpu_ground_truth,
                                             (uint16_t)SVM_CPU_NUM_IMAGES,
                                             &cpu_accuracy,
                                             &cpu_mismatches);
    if (status != XST_SUCCESS) {
        printf("svm_cpu_quantized_eval_accuracy failed: %d\r\n", status);
        return status;
    }

    cpu_accuracy_x1e6 = (uint32_t)(cpu_accuracy * 1000000.0f + 0.5f);
    if (cpu_accuracy_x1e6 > 1000000u) {
        cpu_accuracy_x1e6 = 1000000u;
    }
    cpu_acc_ok = (cpu_accuracy >= TB_MIN_ACCURACY) ? 1 : 0;

    /* Final report: latency/throughput/accuracy (+ optional PMU diagnostics). */
    printf("TB_IMAGES=%u\r\n", (unsigned)MNIST_NUM_IMAGES);
    printf("PL kernel_cycles=%llu kernel_time_us=%llu\r\n",
           (unsigned long long)pl_kernel_cycles,
           (unsigned long long)pl_kernel_time_us);
    printf("PL dma_cycles=%llu dma_time_us=%llu images_per_s=%llu.%03llu\r\n",
           (unsigned long long)pl_dma_cycles,
           (unsigned long long)pl_dma_time_us,
           (unsigned long long)(pl_images_per_s_x1000 / 1000ull),
           (unsigned long long)(pl_images_per_s_x1000 % 1000ull));
    printf("PL mismatches=%u accuracy=%u.%06u threshold=0.98 acc_ok=%d\r\n",
           (unsigned)pl_mismatches,
           (unsigned)(pl_accuracy_x1e6 / 1000000u),
           (unsigned)(pl_accuracy_x1e6 % 1000000u),
           pl_acc_ok);

    printf("PS_QUANTIZED cycles=%llu time_us=%llu images_per_s=%llu.%03llu\r\n",
           (unsigned long long)cpu_cycles,
           (unsigned long long)cpu_time_us,
           (unsigned long long)(cpu_images_per_s_x1000 / 1000ull),
           (unsigned long long)(cpu_images_per_s_x1000 % 1000ull));
    printf("PS_QUANTIZED mismatches=%u accuracy=%u.%06u threshold=0.98 acc_ok=%d\r\n",
           (unsigned)cpu_mismatches,
           (unsigned)(cpu_accuracy_x1e6 / 1000000u),
           (unsigned)(cpu_accuracy_x1e6 % 1000000u),
           cpu_acc_ok);
#if ENABLE_PMU_PROFILING
    printf("PMU cfg11 data_stall=%u i_refill=%u i_tlb_refill=%u\r\n",
           (unsigned)pmu_evt11[0],
           (unsigned)pmu_evt11[1],
           (unsigned)pmu_evt11[2]);
    printf("PMU cfg11 d_refill=%u d_access=%u d_tlb_refill=%u\r\n",
           (unsigned)pmu_evt11[3],
           (unsigned)pmu_evt11[4],
           (unsigned)pmu_evt11[5]);
    printf("PMU derived d_refill_per_1k_access=%llu.%03llu data_stall_per_img=%llu.%03llu\r\n",
           (unsigned long long)(l1d_refill_per_kaccess_x1000 / 1000ull),
           (unsigned long long)(l1d_refill_per_kaccess_x1000 % 1000ull),
           (unsigned long long)(data_stall_per_img_x1000 / 1000ull),
           (unsigned long long)(data_stall_per_img_x1000 % 1000ull));
    printf("L2 read_req=%u read_hit=%u read_hit_rate=%llu.%03llu i_refill_per_img=%llu.%03llu\r\n",
           (unsigned)l2_drreq,
           (unsigned)l2_drhit,
           (unsigned long long)(l2_read_hit_rate_x1000 / 1000ull),
           (unsigned long long)(l2_read_hit_rate_x1000 % 1000ull),
           (unsigned long long)(instr_refill_per_img_x1000 / 1000ull),
           (unsigned long long)(instr_refill_per_img_x1000 % 1000ull));
#if ENABLE_PMU_MULTI_CONFIG
    printf("PMU cfg5 maintlb_stall=%u data_evict=%u nodispatch=%u issueempty=%u\r\n",
           (unsigned)pmu_evt5[0],
           (unsigned)pmu_evt5[3],
           (unsigned)pmu_evt5[4],
           (unsigned)pmu_evt5[5]);
    printf("PMU cfg5 nodispatch_ratio=%llu.%03llu issueempty_ratio=%llu.%03llu\r\n",
           (unsigned long long)(nodispatch_ratio_x1000 / 1000ull),
           (unsigned long long)(nodispatch_ratio_x1000 % 1000ull),
           (unsigned long long)(issueempty_ratio_x1000 / 1000ull),
           (unsigned long long)(issueempty_ratio_x1000 % 1000ull));
    printf("PMU cfg7 neonrename=%u pldstall=%u writestall=%u data_tlbstall=%u\r\n",
           (unsigned)pmu_evt7[0],
           (unsigned)pmu_evt7[1],
           (unsigned)pmu_evt7[2],
           (unsigned)pmu_evt7[4]);
    printf("PMU cfg7 pldstall_ratio=%llu.%03llu writestall_ratio=%llu.%03llu neonrename_per_img=%llu.%03llu\r\n",
           (unsigned long long)(pldstall_ratio_x1000 / 1000ull),
           (unsigned long long)(pldstall_ratio_x1000 % 1000ull),
           (unsigned long long)(writestall_ratio_x1000 / 1000ull),
           (unsigned long long)(writestall_ratio_x1000 % 1000ull),
           (unsigned long long)(neonrename_per_img_x1000 / 1000ull),
           (unsigned long long)(neonrename_per_img_x1000 % 1000ull));
#endif
#endif

    return (pl_acc_ok && cpu_acc_ok) ? 0 : 1;
}
