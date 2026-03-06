#ifndef SVM_PS_DRIVER_H_
#define SVM_PS_DRIVER_H_

#include <stdint.h>

/*
 * Initialize AXI DMA + SVM IP driver instances.
 * Safe to call multiple times; repeated successful calls are harmless.
 */
int svm_init_hw(void);

/*
 * Asynchronously launch one PL batch.
 * - in_q7_1: input buffer, size = 784 * n_images bytes.
 * - out_label: output buffer, size = n_images bytes.
 * - n_images: number of images in this batch.
 */
int svm_run_batch_async_start(const int8_t *in_q7_1,
                              uint8_t *out_label,
                              uint16_t n_images);

/*
 * Wait for a previously started async batch and collect timings.
 * - mm2s_to_s2mm_cycles: MM2S launch -> S2MM done.
 * - kernel_apstart_to_done_cycles: ap_start -> ap_done.
 */
int svm_run_batch_async_wait(uint8_t *out_label,
                             uint16_t n_images,
                             uint64_t *mm2s_to_s2mm_cycles,
                             uint64_t *kernel_apstart_to_done_cycles);

/*
 * Synchronous helper = async_start + async_wait.
 * This is the default path used by current main.c.
 */
int svm_run_batch_timed(const int8_t *in_q7_1,
                        uint8_t *out_label,
                        uint16_t n_images,
                        uint64_t *mm2s_to_s2mm_cycles,
                        uint64_t *kernel_apstart_to_done_cycles);

/*
 * Accuracy-only evaluation (no confusion matrix).
 * Compares bit0 of prediction/ground truth.
 */
int svm_eval_accuracy_only(const uint8_t *pred,
                           const uint8_t *gt,
                           uint16_t n_images,
                           float *acc_out,
                           uint32_t *mismatches_out);

#endif  // SVM_PS_DRIVER_H_
