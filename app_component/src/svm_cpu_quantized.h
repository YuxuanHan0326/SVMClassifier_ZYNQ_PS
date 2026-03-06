#ifndef SVM_CPU_QUANTIZED_H_
#define SVM_CPU_QUANTIZED_H_

#include <stdint.h>

typedef void (*svm_cpu_progress_hook_t)(uint32_t img_idx, void *user);

int svm_cpu_quantized_prepare(void);

void svm_cpu_quantized_set_progress_hook(svm_cpu_progress_hook_t hook,
                                         void *user,
                                         uint32_t period_images);

int svm_cpu_quantized_run_batch_timed(const int8_t *in_q7_1,
                                      uint8_t *out_label,
                                      uint16_t n_images,
                                      uint64_t *cpu_cycles);

int svm_cpu_quantized_eval_accuracy(const uint8_t *pred,
                                    const uint8_t *gt,
                                    uint16_t n_images,
                                    float *acc_out,
                                    uint32_t *mismatches_out);

#endif  // SVM_CPU_QUANTIZED_H_
