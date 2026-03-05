#ifndef MNIST_Q7_1_DATA_H_
#define MNIST_Q7_1_DATA_H_

#include <stdint.h>

#define MNIST_IMG_SIZE 784u
#define MNIST_NUM_IMAGES 2601u
#define MNIST_INPUT_SIZE_BYTES (MNIST_IMG_SIZE * MNIST_NUM_IMAGES)

extern const int8_t g_mnist_test_q7_1[MNIST_INPUT_SIZE_BYTES];
extern const uint8_t g_mnist_ground_truth[MNIST_NUM_IMAGES];

#endif  // MNIST_Q7_1_DATA_H_
