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

static uint8_t g_predictions[MNIST_NUM_IMAGES] __attribute__((aligned(64)));

int main(void) {
    int status;
    uint64_t dma_cycles = 0u;
    uint64_t kernel_cycles = 0u;
    uint64_t dma_time_us = 0u;
    uint64_t kernel_time_us = 0u;
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

    status = svm_run_batch_timed(g_mnist_test_q7_1,
                                 g_predictions,
                                 (uint16_t)MNIST_NUM_IMAGES,
                                 &dma_cycles,
                                 &kernel_cycles);
    if (status != XST_SUCCESS) {
        printf("svm_run_batch_timed failed at N=%u: %d\r\n", (unsigned)MNIST_NUM_IMAGES, status);
        return status;
    }

    kernel_time_us = (kernel_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;
    dma_time_us = (dma_cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;

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
           (unsigned long long)kernel_cycles,
           (unsigned long long)kernel_time_us);
    printf("N=2601 mm2s_to_s2mm_cycles=%llu mm2s_to_s2mm_time_us=%llu\r\n",
           (unsigned long long)dma_cycles,
           (unsigned long long)dma_time_us);
    printf("accuracy=%u.%06u threshold=0.98 acc_ok=%d\r\n",
           (unsigned)(accuracy_x1e6 / 1000000u),
           (unsigned)(accuracy_x1e6 % 1000000u),
           acc_ok);

    return acc_ok ? 0 : 1;
}
