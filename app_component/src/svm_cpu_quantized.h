#ifndef SVM_CPU_QUANTIZED_H_
#define SVM_CPU_QUANTIZED_H_

#include <stdint.h>

int svm_cpu_quantized_prepare(void);

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
