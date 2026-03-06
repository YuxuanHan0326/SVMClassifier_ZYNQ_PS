#ifndef SVM_CPU_MODEL_DATA_H_
#define SVM_CPU_MODEL_DATA_H_

#include <stdint.h>

#define SVM_CPU_IMG_SIZE 784u
#define SVM_CPU_NUM_SV 165u
#define SVM_CPU_NUM_IMAGES 2601u
#define SVM_CPU_GAMMA 0.001f

extern const float g_svm_cpu_svs[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE];
extern const float g_svm_cpu_alphas[SVM_CPU_NUM_SV];
extern const float g_svm_cpu_bias;
extern const float g_svm_cpu_test_data[SVM_CPU_NUM_IMAGES * SVM_CPU_IMG_SIZE];
extern const uint8_t g_svm_cpu_ground_truth[SVM_CPU_NUM_IMAGES];

#endif  // SVM_CPU_MODEL_DATA_H_
