#ifndef SVM_PS_DRIVER_H_
#define SVM_PS_DRIVER_H_

#include <stdint.h>

int svm_init_hw(void);

int svm_run_batch_timed(const int8_t *in_q7_1,
                        uint8_t *out_label,
                        uint16_t n_images,
                        uint64_t *mm2s_to_s2mm_cycles,
                        uint64_t *kernel_apstart_to_done_cycles);

int svm_eval_accuracy_only(const uint8_t *pred,
                           const uint8_t *gt,
                           uint16_t n_images,
                           float *acc_out,
                           uint32_t *mismatches_out);

#endif  // SVM_PS_DRIVER_H_
