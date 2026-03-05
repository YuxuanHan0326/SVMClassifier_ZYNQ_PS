#include <stdint.h>
#include <stdio.h>

#include "mnist_q7_1_data.h"
#include "svm_ps_driver.h"
#include "xparameters.h"
#include "xstatus.h"

#ifndef XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#ifndef XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#include "xtime_l.h"

#define TB_MIN_ACCURACY 0.98f
#define DEBUG_SWEEP_POINTS 7u

static uint8_t g_predictions[MNIST_NUM_IMAGES] __attribute__((aligned(64)));
static const uint16_t k_debug_n_list[DEBUG_SWEEP_POINTS] = {1u, 8u, 32u, 256u, 512u, 1024u, (uint16_t)MNIST_NUM_IMAGES};

int main(void) {
    int status;
    uint64_t dma_cycles = 0u;
    uint64_t kernel_cycles = 0u;
    uint64_t dma_time_us = 0u;
    uint64_t kernel_time_us = 0u;
    uint64_t full_dma_cycles = 0u;
    uint64_t full_kernel_cycles = 0u;
    uint64_t full_dma_time_us = 0u;
    uint64_t full_kernel_time_us = 0u;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    double denom;
    double t0_cycles_d = 0.0;
    double timg_cycles_d = 0.0;
    uint64_t t0_cycles = 0u;
    uint64_t timg_cycles = 0u;
    uint64_t t0_time_us = 0u;
    uint64_t timg_time_us = 0u;
    uint16_t n_images;
    uint32_t i;
    float accuracy = 0.0f;
    uint32_t mismatches = 0u;
    uint32_t accuracy_x1e6;
    int acc_ok;

    printf("Starting Program\r\n");

    status = svm_init_hw();
    if (status != XST_SUCCESS) {
        printf("svm_init_hw failed: %d\r\n", status);
        return status;
    }

    printf("Debug sweep: latency(ap_start->ap_done)\r\n");
    for (i = 0u; i < DEBUG_SWEEP_POINTS; ++i) {
        n_images = k_debug_n_list[i];
        status = svm_run_batch_timed(g_mnist_test_q7_1,
                                     g_predictions,
                                     n_images,
                                     &dma_cycles,
                                     &kernel_cycles);
        if (status != XST_SUCCESS) {
            printf("svm_run_batch_timed failed at N=%u: %d\r\n", (unsigned)n_images, status);
            return status;
        }

        kernel_time_us = (kernel_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
        dma_time_us = (dma_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;

        printf("N=%u kernel_cycles=%llu kernel_time_us=%llu dma_cycles=%llu dma_time_us=%llu\r\n",
               (unsigned)n_images,
               (unsigned long long)kernel_cycles,
               (unsigned long long)kernel_time_us,
               (unsigned long long)dma_cycles,
               (unsigned long long)dma_time_us);

        sum_x += (double)n_images;
        sum_y += (double)kernel_cycles;
        sum_xx += (double)n_images * (double)n_images;
        sum_xy += (double)n_images * (double)kernel_cycles;

        if (n_images == (uint16_t)MNIST_NUM_IMAGES) {
            full_dma_cycles = dma_cycles;
            full_kernel_cycles = kernel_cycles;
            full_dma_time_us = dma_time_us;
            full_kernel_time_us = kernel_time_us;
        }
    }

    denom = ((double)DEBUG_SWEEP_POINTS * sum_xx) - (sum_x * sum_x);
    if (denom != 0.0) {
        timg_cycles_d = (((double)DEBUG_SWEEP_POINTS * sum_xy) - (sum_x * sum_y)) / denom;
        t0_cycles_d = (sum_y - (timg_cycles_d * sum_x)) / (double)DEBUG_SWEEP_POINTS;

        if (t0_cycles_d < 0.0) {
            t0_cycles = 0u;
        } else {
            t0_cycles = (uint64_t)(t0_cycles_d + 0.5);
        }
        if (timg_cycles_d < 0.0) {
            timg_cycles = 0u;
        } else {
            timg_cycles = (uint64_t)(timg_cycles_d + 0.5);
        }

        t0_time_us = (t0_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
        timg_time_us = (timg_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;

        printf("fit_T(N)=T0+N*Timg => T0_cycles=%llu T0_time_us=%llu Timg_cycles=%llu Timg_time_us=%llu\r\n",
               (unsigned long long)t0_cycles,
               (unsigned long long)t0_time_us,
               (unsigned long long)timg_cycles,
               (unsigned long long)timg_time_us);
    } else {
        printf("fit failed: denominator is zero\r\n");
    }

    // Last sweep point is N=2601, so g_predictions now contains full-dataset outputs.
    status = svm_eval_accuracy_only(g_predictions,
                                    g_mnist_ground_truth,
                                    (uint16_t)MNIST_NUM_IMAGES,
                                    &accuracy,
                                    &mismatches);
    if (status != XST_SUCCESS) {
        printf("svm_eval_accuracy_only failed: %d\r\n", status);
        return status;
    }

    accuracy_x1e6 = (uint32_t)(accuracy * 1000000.0f + 0.5f);
    if (accuracy_x1e6 > 1000000u) {
        accuracy_x1e6 = 1000000u;
    }
    acc_ok = (accuracy >= TB_MIN_ACCURACY) ? 1 : 0;

    printf("TB_IMAGES=%u mismatches=%u\r\n", (unsigned)MNIST_NUM_IMAGES, (unsigned)mismatches);
    printf("N=2601 kernel_apstart_to_done_cycles=%llu kernel_apstart_to_done_time_us=%llu\r\n",
           (unsigned long long)full_kernel_cycles,
           (unsigned long long)full_kernel_time_us);
    printf("N=2601 mm2s_to_s2mm_cycles=%llu mm2s_to_s2mm_time_us=%llu\r\n",
           (unsigned long long)full_dma_cycles,
           (unsigned long long)full_dma_time_us);
    printf("accuracy=%u.%06u threshold=0.98 acc_ok=%d\r\n",
           (unsigned)(accuracy_x1e6 / 1000000u),
           (unsigned)(accuracy_x1e6 % 1000000u),
           acc_ok);

    return acc_ok ? 0 : 1;
}
