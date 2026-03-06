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
