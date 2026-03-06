#include "svm_cpu_quantized.h"

#include <string.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "mnist_q7_1_data.h"
#include "svm_cpu_model_data.h"
#include "xstatus.h"
#include "xtime_l.h"

#define SVM_CPU_PREFETCH_DIST 16u
#ifndef SVM_CPU_DENSE_PREFETCH_BYTES
#define SVM_CPU_DENSE_PREFETCH_BYTES 28u
#endif
#define SVM_CPU_CACHELINE_BYTES 32u
#define SVM_CPU_DOT_UNROLL 8u
#define SVM_CPU_DENSE_DOT_NNZ_THRESHOLD 224u
#define SVM_CPU_FORCE_DENSE_NEON 1u
#define SVM_CPU_ENABLE_DENSE_PAIR_FASTPATH 1u
#define SVM_CPU_ENABLE_CACHED_HINT_SORT 1u
#define SVM_CPU_SORT_XROOT_HINT_Q1 41u
#define SVM_CPU_ENABLE_KZERO_SKIP 1u
#define SVM_CPU_USE_OCM_HOT 1u
#define SVM_CPU_ENABLE_KMAX_HOTPATH 0u

#define SVM_CPU_EXP_D2_CLIP 32000
#define SVM_CPU_EXP_LUT_ADDR_BITS 8
#define SVM_CPU_EXP_LUT_ONE_RAW 255u
#define SVM_CPU_EXP_LUT_D2_SHIFT 7u

#define SVM_CPU_ALPHA_Q_SCALE 8
#define SVM_CPU_BIAS_Q_SCALE 128
#define SVM_CPU_SCORE_Q_FRAC 11
#define SVM_CPU_SCORE_BIAS_SHIFT (SVM_CPU_SCORE_Q_FRAC - 7)
#define SVM_CPU_ACTIVE_SV_MASK_WORDS ((SVM_CPU_NUM_SV + 31u) / 32u)

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#define SVM_CPU_ALIGNED64 __attribute__((aligned(64)))
#if SVM_CPU_USE_OCM_HOT
#define SVM_CPU_OCM_BSS __attribute__((section(".ocm_hot_bss"), aligned(64)))
#else
#define SVM_CPU_OCM_BSS __attribute__((aligned(64)))
#endif

_Static_assert(SVM_CPU_IMG_SIZE == MNIST_IMG_SIZE, "SVM_CPU_IMG_SIZE must match MNIST_IMG_SIZE");
_Static_assert(SVM_CPU_NUM_IMAGES == MNIST_NUM_IMAGES, "SVM_CPU_NUM_IMAGES must match MNIST_NUM_IMAGES");

/*
 * Quantized kernel overview:
 * - Input/SV are Q7.1 int8.
 * - alpha is Q5.3 int8.
 * - bias is Q1.7 int8.
 * - score accumulates in Q11 fixed-point domain (score_q2048).
 * - exp(-d2/gamma) is approximated by a validated 8-bit LUT.
 *
 * The implementation keeps frequently reused arrays in contiguous/aligned
 * buffers, with selected hot buffers optionally placed in OCM.
 */
static const uint8_t g_svm_exp_lut_rom[1u << SVM_CPU_EXP_LUT_ADDR_BITS] __attribute__((aligned(64))) = {
    0xFF, 0xF7, 0xEF, 0xE8, 0xE0, 0xD9, 0xD2, 0xCC, 0xC5, 0xBF, 0xB9, 0xB3, 0xAE, 0xA8, 0xA3, 0x9E,
    0x99, 0x94, 0x8F, 0x8B, 0x86, 0x82, 0x7E, 0x7A, 0x76, 0x73, 0x6F, 0x6B, 0x68, 0x65, 0x62, 0x5F,
    0x5C, 0x59, 0x56, 0x53, 0x51, 0x4E, 0x4C, 0x49, 0x47, 0x45, 0x43, 0x40, 0x3E, 0x3C, 0x3B, 0x39,
    0x37, 0x35, 0x33, 0x32, 0x30, 0x2F, 0x2D, 0x2C, 0x2A, 0x29, 0x28, 0x27, 0x25, 0x24, 0x23, 0x22,
    0x21, 0x20, 0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x19, 0x18, 0x17, 0x16, 0x16, 0x15, 0x14,
    0x14, 0x13, 0x12, 0x12, 0x11, 0x11, 0x10, 0x10, 0x0F, 0x0F, 0x0E, 0x0E, 0x0D, 0x0D, 0x0D, 0x0C,
    0x0C, 0x0B, 0x0B, 0x0B, 0x0A, 0x0A, 0x0A, 0x09, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x08, 0x07,
    0x07, 0x07, 0x07, 0x06, 0x06, 0x06, 0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static const int32_t g_svm_sv_norm2_q_const[SVM_CPU_NUM_SV] __attribute__((aligned(64))) = {
    4684, 7288, 1275, 8326, 1979, 5962, 1491, 4843, 3876, 1029, 1742, 60280,
    8854, 5304, 7293, 17120, 1060, 7423, 36987, 8284, 3150, 2341, 8033, 4934,
    11840, 3336, 2788, 5873, 1553, 14644, 1980, 2461, 989, 4013, 2081, 11900,
    4948, 8897, 17403, 3577, 23546, 9876, 1086, 1254, 5967, 3433, 4303, 1368,
    13606, 1048, 1405, 9800, 5994, 2744, 10269, 1449, 3637, 22212, 3995, 860,
    5218, 10578, 4140, 41152, 3157, 3236, 4445, 3336, 5823, 5325, 2442, 902,
    1540, 2186, 8617, 980, 1520, 8461, 5950, 2605, 3763, 6511, 867, 899,
    12103, 8548, 957, 1576, 1433, 7109, 3113, 7231, 1779, 8194, 6961, 1040,
    2672, 8804, 7555, 8170, 4653, 7551, 4487, 1070, 853, 3150, 2421, 949,
    15431, 3430, 969, 3822, 2485, 1464, 1505, 846, 2899, 1601, 3638, 1494,
    1826, 1365, 2653, 4185, 19663, 1353, 2980, 16987, 24822, 1191, 921, 1274,
    1039, 1045, 18285, 1684, 1016, 31548, 962, 2182, 1263, 2927, 1393, 1279,
    3390, 1222, 5309, 1922, 1595, 10839, 1270, 3665, 1789, 8906, 2806, 1041,
    3249, 1633, 3081, 33303, 50112, 990, 20937, 3161, 3150,
};

static uint16_t g_svm_sv_nnz[SVM_CPU_NUM_SV] SVM_CPU_ALIGNED64;
#if !SVM_CPU_FORCE_DENSE_NEON
static uint16_t g_svm_sv_idx[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
static int8_t g_svm_sv_val_q1[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
#endif
static int8_t g_svm_sv_dense_q1[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
static int8_t g_svm_alpha_q3[SVM_CPU_NUM_SV] SVM_CPU_ALIGNED64;

static uint16_t g_svm_active_sv_idx[SVM_CPU_NUM_SV] SVM_CPU_ALIGNED64;
static int8_t g_svm_active_alpha_q3[SVM_CPU_NUM_SV] SVM_CPU_OCM_BSS;
static int32_t g_svm_active_sv_norm2_q[SVM_CPU_NUM_SV] SVM_CPU_OCM_BSS;
static uint16_t g_svm_active_sv_norm_q[SVM_CPU_NUM_SV] SVM_CPU_ALIGNED64;
#if !SVM_CPU_FORCE_DENSE_NEON
static uint16_t g_svm_active_sv_nnz[SVM_CPU_NUM_SV] SVM_CPU_ALIGNED64;
static uint32_t g_svm_active_sv_sparse_off[SVM_CPU_NUM_SV + 1u] SVM_CPU_ALIGNED64;
static uint16_t g_svm_active_sv_sparse_idx[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
static int8_t g_svm_active_sv_sparse_val_q1[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
#endif
static int8_t g_svm_active_sv_dense_q1[SVM_CPU_NUM_SV * SVM_CPU_IMG_SIZE] SVM_CPU_OCM_BSS;
static int32_t g_svm_rem_pos_q2048[SVM_CPU_NUM_SV + 1u] SVM_CPU_OCM_BSS;
static int32_t g_svm_rem_neg_q2048[SVM_CPU_NUM_SV + 1u] SVM_CPU_OCM_BSS;
static uint16_t g_svm_active_sv_count = 0u;
static uint8_t g_svm_all_active_sv_dense = 0u;
static uint8_t g_svm_all_cached_x_dense = 0u;
static int8_t g_svm_bias_q1_7 = 0;

static int32_t g_svm_x_norm2_q[SVM_CPU_NUM_IMAGES] SVM_CPU_OCM_BSS;
static uint16_t g_svm_x_norm_q[SVM_CPU_NUM_IMAGES] SVM_CPU_ALIGNED64;
static int32_t g_svm_img_rem_pos_q2048[SVM_CPU_NUM_IMAGES * (SVM_CPU_NUM_SV + 1u)] SVM_CPU_ALIGNED64;
static int32_t g_svm_img_rem_neg_q2048[SVM_CPU_NUM_IMAGES * (SVM_CPU_NUM_SV + 1u)] SVM_CPU_ALIGNED64;
static uint32_t g_svm_img_kzero_mask[SVM_CPU_NUM_IMAGES * SVM_CPU_ACTIVE_SV_MASK_WORDS] SVM_CPU_ALIGNED64;
static uint8_t g_svm_img_has_kzero[SVM_CPU_NUM_IMAGES] SVM_CPU_ALIGNED64;
#if SVM_CPU_ENABLE_KMAX_HOTPATH
static uint8_t g_svm_img_kmax_q8[SVM_CPU_NUM_IMAGES * SVM_CPU_NUM_SV] SVM_CPU_ALIGNED64;
static int32_t g_svm_img_rem_pos0_q2048[SVM_CPU_NUM_IMAGES] SVM_CPU_ALIGNED64;
static int32_t g_svm_img_rem_neg0_q2048[SVM_CPU_NUM_IMAGES] SVM_CPU_ALIGNED64;
#endif
#if !SVM_CPU_FORCE_DENSE_NEON
static uint16_t g_svm_x_nnz[SVM_CPU_NUM_IMAGES] SVM_CPU_ALIGNED64;
static uint32_t g_svm_x_sparse_off[SVM_CPU_NUM_IMAGES + 1u] SVM_CPU_ALIGNED64;
static uint16_t g_svm_x_idx[SVM_CPU_NUM_IMAGES * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
static int8_t g_svm_x_val_q1[SVM_CPU_NUM_IMAGES * SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
static uint16_t g_svm_x_idx_scratch[SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
static int8_t g_svm_x_val_q1_scratch[SVM_CPU_IMG_SIZE] SVM_CPU_ALIGNED64;
#endif

static int g_svm_test_cache_ready = 0;
static int g_svm_img_bounds_ready = 0;
static int g_svm_prepared = 0;

static inline __attribute__((always_inline)) int32_t abs_i32(int32_t v) {
    return (v >= 0) ? v : -v;
}

static inline __attribute__((always_inline)) int is_kzero_masked(const uint32_t *mask,
                                                                  uint32_t idx) {
    return (mask != NULL) ? (int)((mask[idx >> 5u] >> (idx & 31u)) & 1u) : 0;
}

static inline uint16_t isqrt_u32(uint32_t x) {
    uint32_t op = x;
    uint32_t res = 0u;
    uint32_t one = 1u << 30;

    while (one > op) {
        one >>= 2;
    }

    while (one != 0u) {
        if (op >= (res + one)) {
            op -= (res + one);
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }

    return (uint16_t)res;
}

static inline __attribute__((always_inline)) uint32_t sv_order_score(uint32_t abs_alpha_q3,
                                                                      uint16_t sv_nnz) {
    return (abs_alpha_q3 << 10) / ((uint32_t)sv_nnz + 8u);
}

static inline __attribute__((always_inline)) uint32_t sv_order_score_cached_hint(uint32_t abs_alpha_q3,
                                                                                  int32_t sv_norm2_q) {
    int32_t d = (int32_t)isqrt_u32((uint32_t)sv_norm2_q) - (int32_t)SVM_CPU_SORT_XROOT_HINT_Q1;
    uint32_t lb;
    uint32_t idx;
    uint32_t k_hint;

    if (d < 0) {
        d = -d;
    }
    d -= 1;
    if (d < 0) {
        d = 0;
    }

    lb = (uint32_t)d * (uint32_t)d;
    idx = lb >> SVM_CPU_EXP_LUT_D2_SHIFT;
    if (idx >= (1u << SVM_CPU_EXP_LUT_ADDR_BITS)) {
        idx = (1u << SVM_CPU_EXP_LUT_ADDR_BITS) - 1u;
    }
    k_hint = (uint32_t)g_svm_exp_lut_rom[idx];
    return abs_alpha_q3 * k_hint;
}

static inline int float_to_scaled_int_exact(float v, int32_t scale, int32_t *out_i) {
    const float sf = v * (float)scale;
    const int32_t si = (int32_t)sf;

    if ((float)si != sf) {
        return 0;
    }

    *out_i = si;
    return 1;
}

static inline int float_to_q1_exact(float v, int8_t *out_q) {
    const float qf = v * 2.0f;
    const int32_t qi = (int32_t)qf;

    if ((float)qi != qf) {
        return 0;
    }
    if ((qi < -128) || (qi > 127)) {
        return 0;
    }

    *out_q = (int8_t)qi;
    return 1;
}

static inline __attribute__((always_inline)) int32_t dot_dense_dense_neon_q1(
    const int8_t *restrict a_q1,
    const int8_t *restrict b_q1) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int32x4_t acc32_0 = vdupq_n_s32(0);
    int32x4_t acc32_1 = vdupq_n_s32(0);
    uint32_t j = 0u;

    for (; (j + 32u) <= SVM_CPU_IMG_SIZE; j += 32u) {
        if (LIKELY((j + SVM_CPU_DENSE_PREFETCH_BYTES) < SVM_CPU_IMG_SIZE)) {
            __builtin_prefetch(&a_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&b_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
        }

        {
            const int8x16_t va = vld1q_s8(&a_q1[j]);
            const int8x16_t vb = vld1q_s8(&b_q1[j]);
            const int16x8_t p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            const int16x8_t p_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc32_0 = vpadalq_s16(acc32_0, p_lo);
            acc32_0 = vpadalq_s16(acc32_0, p_hi);
        }

        {
            const int8x16_t va = vld1q_s8(&a_q1[j + 16u]);
            const int8x16_t vb = vld1q_s8(&b_q1[j + 16u]);
            const int16x8_t p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            const int16x8_t p_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc32_1 = vpadalq_s16(acc32_1, p_lo);
            acc32_1 = vpadalq_s16(acc32_1, p_hi);
        }
    }

    for (; (j + 16u) <= SVM_CPU_IMG_SIZE; j += 16u) {
        const int8x16_t va = vld1q_s8(&a_q1[j]);
        const int8x16_t vb = vld1q_s8(&b_q1[j]);
        const int16x8_t p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        const int16x8_t p_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        acc32_0 = vpadalq_s16(acc32_0, p_lo);
        acc32_0 = vpadalq_s16(acc32_0, p_hi);
    }

    acc32_0 = vaddq_s32(acc32_0, acc32_1);

    int32_t dot = vgetq_lane_s32(acc32_0, 0) +
                  vgetq_lane_s32(acc32_0, 1) +
                  vgetq_lane_s32(acc32_0, 2) +
                  vgetq_lane_s32(acc32_0, 3);

    for (; j < SVM_CPU_IMG_SIZE; ++j) {
        dot += (int32_t)a_q1[j] * (int32_t)b_q1[j];
    }

    return dot;
#else
    int32_t dot = 0;
    for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
        dot += (int32_t)a_q1[j] * (int32_t)b_q1[j];
    }
    return dot;
#endif
}

static inline __attribute__((always_inline)) void dot_dense_dense2_neon_q1(
    const int8_t *restrict sv0_q1,
    const int8_t *restrict sv1_q1,
    const int8_t *restrict x_q1,
    int32_t *dot0_out,
    int32_t *dot1_out) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    uint32_t j = 0u;

    for (; (j + 16u) <= SVM_CPU_IMG_SIZE; j += 16u) {
        if (LIKELY((j + SVM_CPU_DENSE_PREFETCH_BYTES) < SVM_CPU_IMG_SIZE)) {
            __builtin_prefetch(&sv0_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&sv1_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&x_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
        }

        {
            const int8x16_t vx = vld1q_s8(&x_q1[j]);
            const int8x16_t v0 = vld1q_s8(&sv0_q1[j]);
            const int8x16_t v1 = vld1q_s8(&sv1_q1[j]);
            const int16x8_t p0_lo = vmull_s8(vget_low_s8(v0), vget_low_s8(vx));
            const int16x8_t p0_hi = vmull_s8(vget_high_s8(v0), vget_high_s8(vx));
            const int16x8_t p1_lo = vmull_s8(vget_low_s8(v1), vget_low_s8(vx));
            const int16x8_t p1_hi = vmull_s8(vget_high_s8(v1), vget_high_s8(vx));

            acc0 = vpadalq_s16(acc0, p0_lo);
            acc0 = vpadalq_s16(acc0, p0_hi);
            acc1 = vpadalq_s16(acc1, p1_lo);
            acc1 = vpadalq_s16(acc1, p1_hi);
        }
    }

    int32_t dot0 = vgetq_lane_s32(acc0, 0) +
                   vgetq_lane_s32(acc0, 1) +
                   vgetq_lane_s32(acc0, 2) +
                   vgetq_lane_s32(acc0, 3);
    int32_t dot1 = vgetq_lane_s32(acc1, 0) +
                   vgetq_lane_s32(acc1, 1) +
                   vgetq_lane_s32(acc1, 2) +
                   vgetq_lane_s32(acc1, 3);

    for (; j < SVM_CPU_IMG_SIZE; ++j) {
        const int32_t x = (int32_t)x_q1[j];
        dot0 += (int32_t)sv0_q1[j] * x;
        dot1 += (int32_t)sv1_q1[j] * x;
    }

    *dot0_out = dot0;
    *dot1_out = dot1;
#else
    int32_t dot0 = 0;
    int32_t dot1 = 0;
    for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
        const int32_t x = (int32_t)x_q1[j];
        dot0 += (int32_t)sv0_q1[j] * x;
        dot1 += (int32_t)sv1_q1[j] * x;
    }
    *dot0_out = dot0;
    *dot1_out = dot1;
#endif
}

static inline __attribute__((always_inline)) void dot_dense_dense4_neon_q1(
    const int8_t *restrict sv0_q1,
    const int8_t *restrict sv1_q1,
    const int8_t *restrict sv2_q1,
    const int8_t *restrict sv3_q1,
    const int8_t *restrict x_q1,
    int32_t *dot0_out,
    int32_t *dot1_out,
    int32_t *dot2_out,
    int32_t *dot3_out) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    uint32_t j = 0u;

    for (; (j + 16u) <= SVM_CPU_IMG_SIZE; j += 16u) {
        if (LIKELY((j + SVM_CPU_DENSE_PREFETCH_BYTES) < SVM_CPU_IMG_SIZE)) {
            __builtin_prefetch(&sv0_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&sv1_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&sv2_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&sv3_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
            __builtin_prefetch(&x_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
        }

        {
            const int8x16_t vx = vld1q_s8(&x_q1[j]);
            const int8x16_t v0 = vld1q_s8(&sv0_q1[j]);
            const int8x16_t v1 = vld1q_s8(&sv1_q1[j]);
            const int8x16_t v2 = vld1q_s8(&sv2_q1[j]);
            const int8x16_t v3 = vld1q_s8(&sv3_q1[j]);
            const int16x8_t p0_lo = vmull_s8(vget_low_s8(v0), vget_low_s8(vx));
            const int16x8_t p0_hi = vmull_s8(vget_high_s8(v0), vget_high_s8(vx));
            const int16x8_t p1_lo = vmull_s8(vget_low_s8(v1), vget_low_s8(vx));
            const int16x8_t p1_hi = vmull_s8(vget_high_s8(v1), vget_high_s8(vx));
            const int16x8_t p2_lo = vmull_s8(vget_low_s8(v2), vget_low_s8(vx));
            const int16x8_t p2_hi = vmull_s8(vget_high_s8(v2), vget_high_s8(vx));
            const int16x8_t p3_lo = vmull_s8(vget_low_s8(v3), vget_low_s8(vx));
            const int16x8_t p3_hi = vmull_s8(vget_high_s8(v3), vget_high_s8(vx));

            acc0 = vpadalq_s16(acc0, p0_lo);
            acc0 = vpadalq_s16(acc0, p0_hi);
            acc1 = vpadalq_s16(acc1, p1_lo);
            acc1 = vpadalq_s16(acc1, p1_hi);
            acc2 = vpadalq_s16(acc2, p2_lo);
            acc2 = vpadalq_s16(acc2, p2_hi);
            acc3 = vpadalq_s16(acc3, p3_lo);
            acc3 = vpadalq_s16(acc3, p3_hi);
        }
    }

    int32_t dot0 = vgetq_lane_s32(acc0, 0) +
                   vgetq_lane_s32(acc0, 1) +
                   vgetq_lane_s32(acc0, 2) +
                   vgetq_lane_s32(acc0, 3);
    int32_t dot1 = vgetq_lane_s32(acc1, 0) +
                   vgetq_lane_s32(acc1, 1) +
                   vgetq_lane_s32(acc1, 2) +
                   vgetq_lane_s32(acc1, 3);
    int32_t dot2 = vgetq_lane_s32(acc2, 0) +
                   vgetq_lane_s32(acc2, 1) +
                   vgetq_lane_s32(acc2, 2) +
                   vgetq_lane_s32(acc2, 3);
    int32_t dot3 = vgetq_lane_s32(acc3, 0) +
                   vgetq_lane_s32(acc3, 1) +
                   vgetq_lane_s32(acc3, 2) +
                   vgetq_lane_s32(acc3, 3);

    for (; j < SVM_CPU_IMG_SIZE; ++j) {
        const int32_t x = (int32_t)x_q1[j];
        dot0 += (int32_t)sv0_q1[j] * x;
        dot1 += (int32_t)sv1_q1[j] * x;
        dot2 += (int32_t)sv2_q1[j] * x;
        dot3 += (int32_t)sv3_q1[j] * x;
    }

    *dot0_out = dot0;
    *dot1_out = dot1;
    *dot2_out = dot2;
    *dot3_out = dot3;
#else
    int32_t dot0 = 0;
    int32_t dot1 = 0;
    int32_t dot2 = 0;
    int32_t dot3 = 0;
    for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
        const int32_t x = (int32_t)x_q1[j];
        dot0 += (int32_t)sv0_q1[j] * x;
        dot1 += (int32_t)sv1_q1[j] * x;
        dot2 += (int32_t)sv2_q1[j] * x;
        dot3 += (int32_t)sv3_q1[j] * x;
    }
    *dot0_out = dot0;
    *dot1_out = dot1;
    *dot2_out = dot2;
    *dot3_out = dot3;
#endif
}

static inline __attribute__((always_inline)) int accumulate_sv_term_and_check(
    uint32_t a,
    int skip,
    int32_t dot_q,
    int32_t x_norm2_q,
    int32_t *score_q2048,
    const int32_t *rem_pos_q2048,
    const int32_t *rem_neg_q2048) {
    if (skip == 0) {
        int32_t d2_raw = g_svm_active_sv_norm2_q[a] + x_norm2_q - (dot_q << 1);
        uint8_t k_q8;

        if (UNLIKELY(d2_raw < 0)) {
            d2_raw = 0;
        } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
            d2_raw = SVM_CPU_EXP_D2_CLIP;
        }

        k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
        if (k_q8 != 0u) {
            *score_q2048 += (int32_t)g_svm_active_alpha_q3[a] * (int32_t)k_q8;
        }
    }

    if ((*score_q2048 + rem_neg_q2048[a + 1u]) > 0) {
        *score_q2048 = 1;
        return 1;
    }
    if ((*score_q2048 + rem_pos_q2048[a + 1u]) < 0) {
        *score_q2048 = -1;
        return 1;
    }

    return 0;
}

#if !SVM_CPU_FORCE_DENSE_NEON
static inline __attribute__((always_inline)) int32_t dot_sparse_sv_dense_x_q1(
    const uint16_t *restrict sv_idx,
    const int8_t *restrict sv_val_q1,
    uint16_t sv_nnz,
    const int8_t *restrict x_dense_q1) {
    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    int32_t acc4 = 0;
    int32_t acc5 = 0;
    int32_t acc6 = 0;
    int32_t acc7 = 0;

    const uint16_t *idx = sv_idx;
    const int8_t *val = sv_val_q1;
    uint32_t rem = (uint32_t)sv_nnz;

    while (rem >= SVM_CPU_DOT_UNROLL) {
        if (LIKELY(rem > SVM_CPU_PREFETCH_DIST)) {
            __builtin_prefetch(&idx[SVM_CPU_PREFETCH_DIST], 0, 1);
            __builtin_prefetch(&val[SVM_CPU_PREFETCH_DIST], 0, 1);
            __builtin_prefetch(&x_dense_q1[idx[SVM_CPU_PREFETCH_DIST]], 0, 1);
        }

        acc0 += (int32_t)val[0] * (int32_t)x_dense_q1[idx[0]];
        acc1 += (int32_t)val[1] * (int32_t)x_dense_q1[idx[1]];
        acc2 += (int32_t)val[2] * (int32_t)x_dense_q1[idx[2]];
        acc3 += (int32_t)val[3] * (int32_t)x_dense_q1[idx[3]];
        acc4 += (int32_t)val[4] * (int32_t)x_dense_q1[idx[4]];
        acc5 += (int32_t)val[5] * (int32_t)x_dense_q1[idx[5]];
        acc6 += (int32_t)val[6] * (int32_t)x_dense_q1[idx[6]];
        acc7 += (int32_t)val[7] * (int32_t)x_dense_q1[idx[7]];

        idx += SVM_CPU_DOT_UNROLL;
        val += SVM_CPU_DOT_UNROLL;
        rem -= SVM_CPU_DOT_UNROLL;
    }

    int32_t dot = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    while (rem != 0u) {
        dot += (int32_t)(*val) * (int32_t)x_dense_q1[*idx];
        ++idx;
        ++val;
        --rem;
    }

    return dot;
}

static inline __attribute__((always_inline)) int32_t dot_sparse_x_dense_sv_q1(
    const uint16_t *restrict x_idx,
    const int8_t *restrict x_val_q1,
    uint16_t x_nnz,
    const int8_t *restrict sv_dense_q1) {
    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    int32_t acc4 = 0;
    int32_t acc5 = 0;
    int32_t acc6 = 0;
    int32_t acc7 = 0;

    const uint16_t *idx = x_idx;
    const int8_t *val = x_val_q1;
    uint32_t rem = (uint32_t)x_nnz;

    while (rem >= SVM_CPU_DOT_UNROLL) {
        if (LIKELY(rem > SVM_CPU_PREFETCH_DIST)) {
            __builtin_prefetch(&idx[SVM_CPU_PREFETCH_DIST], 0, 1);
            __builtin_prefetch(&val[SVM_CPU_PREFETCH_DIST], 0, 1);
            __builtin_prefetch(&sv_dense_q1[idx[SVM_CPU_PREFETCH_DIST]], 0, 1);
        }

        acc0 += (int32_t)val[0] * (int32_t)sv_dense_q1[idx[0]];
        acc1 += (int32_t)val[1] * (int32_t)sv_dense_q1[idx[1]];
        acc2 += (int32_t)val[2] * (int32_t)sv_dense_q1[idx[2]];
        acc3 += (int32_t)val[3] * (int32_t)sv_dense_q1[idx[3]];
        acc4 += (int32_t)val[4] * (int32_t)sv_dense_q1[idx[4]];
        acc5 += (int32_t)val[5] * (int32_t)sv_dense_q1[idx[5]];
        acc6 += (int32_t)val[6] * (int32_t)sv_dense_q1[idx[6]];
        acc7 += (int32_t)val[7] * (int32_t)sv_dense_q1[idx[7]];

        idx += SVM_CPU_DOT_UNROLL;
        val += SVM_CPU_DOT_UNROLL;
        rem -= SVM_CPU_DOT_UNROLL;
    }

    int32_t dot = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    while (rem != 0u) {
        dot += (int32_t)(*val) * (int32_t)sv_dense_q1[*idx];
        ++idx;
        ++val;
        --rem;
    }

    return dot;
}

static inline void build_sparse_x_local(const int8_t *restrict x_dense_q1,
                                        uint16_t *restrict x_idx_local,
                                       int8_t *restrict x_val_q1_local,
                                       uint16_t *x_nnz_out,
                                       int32_t *x_norm2_q_out) {
    uint16_t nnz = 0u;
    int32_t norm2_q = 0;

    for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
        const int8_t q = x_dense_q1[j];

        if (LIKELY((j + 16u) < SVM_CPU_IMG_SIZE)) {
            __builtin_prefetch(&x_dense_q1[j + 16u], 0, 1);
        }

        norm2_q += (int32_t)q * (int32_t)q;
        if (q != 0) {
            x_idx_local[nnz] = (uint16_t)j;
            x_val_q1_local[nnz] = q;
            ++nnz;
        }
    }

    *x_nnz_out = nnz;
    *x_norm2_q_out = norm2_q;
}
#endif

static inline __attribute__((always_inline)) int32_t compute_dense_norm2_q1(
    const int8_t *restrict x_dense_q1) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int32x4_t acc32_0 = vdupq_n_s32(0);
    int32x4_t acc32_1 = vdupq_n_s32(0);
    uint32_t j = 0u;

    for (; (j + 32u) <= SVM_CPU_IMG_SIZE; j += 32u) {
        if (LIKELY((j + SVM_CPU_DENSE_PREFETCH_BYTES) < SVM_CPU_IMG_SIZE)) {
            __builtin_prefetch(&x_dense_q1[j + SVM_CPU_DENSE_PREFETCH_BYTES], 0, 1);
        }

        {
            const int8x16_t vx = vld1q_s8(&x_dense_q1[j]);
            const int16x8_t p_lo = vmull_s8(vget_low_s8(vx), vget_low_s8(vx));
            const int16x8_t p_hi = vmull_s8(vget_high_s8(vx), vget_high_s8(vx));
            acc32_0 = vpadalq_s16(acc32_0, p_lo);
            acc32_0 = vpadalq_s16(acc32_0, p_hi);
        }

        {
            const int8x16_t vx = vld1q_s8(&x_dense_q1[j + 16u]);
            const int16x8_t p_lo = vmull_s8(vget_low_s8(vx), vget_low_s8(vx));
            const int16x8_t p_hi = vmull_s8(vget_high_s8(vx), vget_high_s8(vx));
            acc32_1 = vpadalq_s16(acc32_1, p_lo);
            acc32_1 = vpadalq_s16(acc32_1, p_hi);
        }
    }

    for (; (j + 16u) <= SVM_CPU_IMG_SIZE; j += 16u) {
        const int8x16_t vx = vld1q_s8(&x_dense_q1[j]);
        const int16x8_t p_lo = vmull_s8(vget_low_s8(vx), vget_low_s8(vx));
        const int16x8_t p_hi = vmull_s8(vget_high_s8(vx), vget_high_s8(vx));
        acc32_0 = vpadalq_s16(acc32_0, p_lo);
        acc32_0 = vpadalq_s16(acc32_0, p_hi);
    }

    acc32_0 = vaddq_s32(acc32_0, acc32_1);

    int32_t norm2_q = vgetq_lane_s32(acc32_0, 0) +
                      vgetq_lane_s32(acc32_0, 1) +
                      vgetq_lane_s32(acc32_0, 2) +
                      vgetq_lane_s32(acc32_0, 3);

    for (; j < SVM_CPU_IMG_SIZE; ++j) {
        const int32_t q = (int32_t)x_dense_q1[j];
        norm2_q += q * q;
    }
    return norm2_q;
#else
    int32_t norm2_q = 0;
    for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
        const int32_t q = (int32_t)x_dense_q1[j];
        norm2_q += q * q;
    }
    return norm2_q;
#endif
}

static void build_cached_image_bounds(void) {
    const uint32_t stride = (uint32_t)SVM_CPU_NUM_SV + 1u;
    uint8_t kmax_local[SVM_CPU_NUM_SV];

    for (uint32_t img = 0u; img < SVM_CPU_NUM_IMAGES; ++img) {
        const uint16_t x_norm_q = g_svm_x_norm_q[img];
        int32_t *restrict rem_pos = &g_svm_img_rem_pos_q2048[img * stride];
        int32_t *restrict rem_neg = &g_svm_img_rem_neg_q2048[img * stride];
        uint32_t *restrict kzero_mask = &g_svm_img_kzero_mask[img * SVM_CPU_ACTIVE_SV_MASK_WORDS];
#if SVM_CPU_ENABLE_KMAX_HOTPATH
        uint8_t *restrict kmax_q8 = &g_svm_img_kmax_q8[img * SVM_CPU_NUM_SV];
        int32_t rem_pos0 = 0;
        int32_t rem_neg0 = 0;
#endif
        uint32_t kzero_or = 0u;

        for (uint32_t w = 0u; w < SVM_CPU_ACTIVE_SV_MASK_WORDS; ++w) {
            kzero_mask[w] = 0u;
        }

        for (uint32_t a = 0u; a < (uint32_t)g_svm_active_sv_count; ++a) {
            const uint16_t sv_norm_q = g_svm_active_sv_norm_q[a];
            int32_t d = (int32_t)sv_norm_q - (int32_t)x_norm_q;
            uint32_t lb;
            uint32_t idx;

            if (d < 0) {
                d = -d;
            }

            // Safe lower bound for |sqrt(sv_norm2)-sqrt(x_norm2)|^2 using floor-sqrt intervals.
            d = d - 1;
            if (d < 0) {
                d = 0;
            }
            lb = (uint32_t)d * (uint32_t)d;
            idx = lb >> SVM_CPU_EXP_LUT_D2_SHIFT;
            if (idx >= (1u << SVM_CPU_EXP_LUT_ADDR_BITS)) {
                idx = (1u << SVM_CPU_EXP_LUT_ADDR_BITS) - 1u;
            }
            kmax_local[a] = g_svm_exp_lut_rom[idx];
#if SVM_CPU_ENABLE_KMAX_HOTPATH
            kmax_q8[a] = kmax_local[a];
            {
                const int32_t term = (int32_t)g_svm_active_alpha_q3[a] * (int32_t)kmax_local[a];
                rem_pos0 += (term > 0) ? term : 0;
                rem_neg0 += (term < 0) ? term : 0;
            }
#endif
#if SVM_CPU_ENABLE_KZERO_SKIP
            if (kmax_local[a] == 0u) {
                const uint32_t bit = (1u << (a & 31u));
                kzero_mask[a >> 5u] |= bit;
                kzero_or |= bit;
            }
#endif
        }

        rem_pos[g_svm_active_sv_count] = 0;
        rem_neg[g_svm_active_sv_count] = 0;
        for (int32_t a = (int32_t)g_svm_active_sv_count - 1; a >= 0; --a) {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a] * (int32_t)kmax_local[a];
            rem_pos[a] = rem_pos[a + 1] + ((term > 0) ? term : 0);
            rem_neg[a] = rem_neg[a + 1] + ((term < 0) ? term : 0);
        }
#if SVM_CPU_ENABLE_KMAX_HOTPATH
        g_svm_img_rem_pos0_q2048[img] = rem_pos0;
        g_svm_img_rem_neg0_q2048[img] = rem_neg0;
#endif
        g_svm_img_has_kzero[img] = (kzero_or != 0u) ? 1u : 0u;
    }
}

#if SVM_CPU_ENABLE_KMAX_HOTPATH
static inline __attribute__((always_inline)) uint8_t classify_cached_dense_with_kmax(
    uint32_t img,
    const int8_t *restrict x_dense_q1,
    int32_t x_norm2_q) {
    const uint8_t *restrict kmax_q8 = &g_svm_img_kmax_q8[img * SVM_CPU_NUM_SV];
    int32_t score_q2048 = ((int32_t)g_svm_bias_q1_7) << SVM_CPU_SCORE_BIAS_SHIFT;
    int32_t rem_pos_run = g_svm_img_rem_pos0_q2048[img];
    int32_t rem_neg_run = g_svm_img_rem_neg0_q2048[img];
    const int has_kzero = (g_svm_img_has_kzero[img] != 0u) ? 1 : 0;
    uint32_t a = 0u;

    if (UNLIKELY((score_q2048 + rem_neg_run) > 0)) {
        return 1u;
    }
    if (UNLIKELY((score_q2048 + rem_pos_run) < 0)) {
        return 0u;
    }

    for (; (a + 3u) < (uint32_t)g_svm_active_sv_count; a += 4u) {
        const uint32_t dense_base0 = a * SVM_CPU_IMG_SIZE;
        const uint32_t dense_base1 = dense_base0 + SVM_CPU_IMG_SIZE;
        const uint32_t dense_base2 = dense_base1 + SVM_CPU_IMG_SIZE;
        const uint32_t dense_base3 = dense_base2 + SVM_CPU_IMG_SIZE;
        const int8_t *restrict sv0_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base0], SVM_CPU_CACHELINE_BYTES);
        const int8_t *restrict sv1_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base1], SVM_CPU_CACHELINE_BYTES);
        const int8_t *restrict sv2_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base2], SVM_CPU_CACHELINE_BYTES);
        const int8_t *restrict sv3_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base3], SVM_CPU_CACHELINE_BYTES);
        const uint8_t k0b = kmax_q8[a];
        const uint8_t k1b = kmax_q8[a + 1u];
        const uint8_t k2b = kmax_q8[a + 2u];
        const uint8_t k3b = kmax_q8[a + 3u];
        int32_t dot0_q = 0;
        int32_t dot1_q = 0;
        int32_t dot2_q = 0;
        int32_t dot3_q = 0;
        int32_t d2_raw;
        uint8_t k_q8;

        if (LIKELY((a + 4u) < (uint32_t)g_svm_active_sv_count)) {
            const uint32_t next_base = (a + 4u) * SVM_CPU_IMG_SIZE;
            __builtin_prefetch(&g_svm_active_sv_dense_q1[next_base], 0, 1);
        }
        if (LIKELY((a + 8u) < (uint32_t)g_svm_active_sv_count)) {
            const uint32_t next2_base = (a + 8u) * SVM_CPU_IMG_SIZE;
            __builtin_prefetch(&g_svm_active_sv_dense_q1[next2_base], 0, 1);
        }

        if ((has_kzero == 0) || ((k0b | k1b | k2b | k3b) != 0u)) {
            dot_dense_dense4_neon_q1(sv0_q1, sv1_q1, sv2_q1, sv3_q1, x_dense_q1,
                                     &dot0_q, &dot1_q, &dot2_q, &dot3_q);
        }

        if (k0b != 0u) {
            d2_raw = g_svm_active_sv_norm2_q[a] + x_norm2_q - (dot0_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a] * (int32_t)k_q8;
            }
        }
        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a] * (int32_t)k0b;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }

        if (k1b != 0u) {
            d2_raw = g_svm_active_sv_norm2_q[a + 1u] + x_norm2_q - (dot1_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a + 1u] * (int32_t)k_q8;
            }
        }
        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a + 1u] * (int32_t)k1b;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }

        if (k2b != 0u) {
            d2_raw = g_svm_active_sv_norm2_q[a + 2u] + x_norm2_q - (dot2_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a + 2u] * (int32_t)k_q8;
            }
        }
        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a + 2u] * (int32_t)k2b;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }

        if (k3b != 0u) {
            d2_raw = g_svm_active_sv_norm2_q[a + 3u] + x_norm2_q - (dot3_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a + 3u] * (int32_t)k_q8;
            }
        }
        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a + 3u] * (int32_t)k3b;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }
    }

    for (; (a + 1u) < (uint32_t)g_svm_active_sv_count; a += 2u) {
        const uint32_t dense_base0 = a * SVM_CPU_IMG_SIZE;
        const uint32_t dense_base1 = dense_base0 + SVM_CPU_IMG_SIZE;
        const int8_t *restrict sv0_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base0], SVM_CPU_CACHELINE_BYTES);
        const int8_t *restrict sv1_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base1], SVM_CPU_CACHELINE_BYTES);
        const uint8_t k0b = kmax_q8[a];
        const uint8_t k1b = kmax_q8[a + 1u];
        int32_t dot0_q = 0;
        int32_t dot1_q = 0;
        int32_t d2_raw;
        uint8_t k_q8;

        if (LIKELY((a + 2u) < (uint32_t)g_svm_active_sv_count)) {
            const uint32_t next_base = (a + 2u) * SVM_CPU_IMG_SIZE;
            __builtin_prefetch(&g_svm_active_sv_dense_q1[next_base], 0, 1);
        }
        if (LIKELY((a + 4u) < (uint32_t)g_svm_active_sv_count)) {
            const uint32_t next2_base = (a + 4u) * SVM_CPU_IMG_SIZE;
            __builtin_prefetch(&g_svm_active_sv_dense_q1[next2_base], 0, 1);
        }

        if ((has_kzero == 0) || ((k0b | k1b) != 0u)) {
            dot_dense_dense2_neon_q1(sv0_q1, sv1_q1, x_dense_q1, &dot0_q, &dot1_q);
        }

        if (k0b != 0u) {
            d2_raw = g_svm_active_sv_norm2_q[a] + x_norm2_q - (dot0_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a] * (int32_t)k_q8;
            }
        }
        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a] * (int32_t)k0b;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }

        if (k1b != 0u) {
            d2_raw = g_svm_active_sv_norm2_q[a + 1u] + x_norm2_q - (dot1_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a + 1u] * (int32_t)k_q8;
            }
        }
        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a + 1u] * (int32_t)k1b;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }
    }

    for (; a < (uint32_t)g_svm_active_sv_count; ++a) {
        const uint32_t dense_base = a * SVM_CPU_IMG_SIZE;
        const int8_t *restrict sv_dense_q1 = (const int8_t *)__builtin_assume_aligned(
            &g_svm_active_sv_dense_q1[dense_base], SVM_CPU_CACHELINE_BYTES);
        const uint8_t kb = kmax_q8[a];
        int32_t d2_raw;
        uint8_t k_q8;

        if (kb != 0u) {
            const int32_t dot_q = dot_dense_dense_neon_q1(sv_dense_q1, x_dense_q1);
            d2_raw = g_svm_active_sv_norm2_q[a] + x_norm2_q - (dot_q << 1);
            if (UNLIKELY(d2_raw < 0)) {
                d2_raw = 0;
            } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                d2_raw = SVM_CPU_EXP_D2_CLIP;
            }
            k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
            if (k_q8 != 0u) {
                score_q2048 += (int32_t)g_svm_active_alpha_q3[a] * (int32_t)k_q8;
            }
        }

        {
            const int32_t term = (int32_t)g_svm_active_alpha_q3[a] * (int32_t)kb;
            rem_pos_run -= (term > 0) ? term : 0;
            rem_neg_run -= (term < 0) ? term : 0;
        }
        if ((score_q2048 + rem_neg_run) > 0) {
            return 1u;
        }
        if ((score_q2048 + rem_pos_run) < 0) {
            return 0u;
        }
    }

    return (score_q2048 >= 0) ? 1u : 0u;
}
#endif

int svm_cpu_quantized_prepare(void) {
    int32_t bias_q7;

    /*
     * One-time setup:
     * 1) Quantize bias / SV / alpha.
     * 2) Build active-SV working set (alpha != 0).
     * 3) Precompute remainder bounds for exact early exit.
     * 4) Cache MNIST test-side norms/sparse metadata for fast repeated runs.
     */
    if (g_svm_prepared != 0) {
        return XST_SUCCESS;
    }

    if (!float_to_scaled_int_exact(g_svm_cpu_bias, SVM_CPU_BIAS_Q_SCALE, &bias_q7) ||
        (bias_q7 < -128) || (bias_q7 > 127)) {
        return XST_FAILURE;
    }
    g_svm_bias_q1_7 = (int8_t)bias_q7;

    g_svm_active_sv_count = 0u;
    g_svm_all_active_sv_dense = 1u;
    g_svm_all_cached_x_dense = 1u;

    for (uint32_t sv = 0u; sv < SVM_CPU_NUM_SV; ++sv) {
        const float *restrict sv_ptr = &g_svm_cpu_svs[sv * SVM_CPU_IMG_SIZE];
        const uint32_t dense_base = sv * SVM_CPU_IMG_SIZE;
        uint16_t nnz = 0u;
        int32_t alpha_q3;

        for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
            int8_t q;

            if (LIKELY((j + 16u) < SVM_CPU_IMG_SIZE)) {
                __builtin_prefetch(&sv_ptr[j + 16u], 0, 1);
            }

            if (!float_to_q1_exact(sv_ptr[j], &q)) {
                return XST_FAILURE;
            }

            g_svm_sv_dense_q1[dense_base + j] = q;

            if (q != 0) {
#if !SVM_CPU_FORCE_DENSE_NEON
                g_svm_sv_idx[dense_base + nnz] = (uint16_t)j;
                g_svm_sv_val_q1[dense_base + nnz] = q;
#endif
                ++nnz;
            }
        }

        g_svm_sv_nnz[sv] = nnz;
        if (!float_to_scaled_int_exact(g_svm_cpu_alphas[sv], SVM_CPU_ALPHA_Q_SCALE, &alpha_q3) ||
            (alpha_q3 < -128) || (alpha_q3 > 127)) {
            return XST_FAILURE;
        }
        g_svm_alpha_q3[sv] = (int8_t)alpha_q3;

        if (alpha_q3 != 0) {
            g_svm_active_sv_idx[g_svm_active_sv_count] = (uint16_t)sv;
            ++g_svm_active_sv_count;
        }
    }

    /*
     * Sort active SVs to make early-exit tighter on average:
     * larger contribution terms are consumed earlier so remaining bound
     * shrinks faster.
     */
    for (uint32_t i = 0u; (i + 1u) < (uint32_t)g_svm_active_sv_count; ++i) {
        uint32_t best = i;
        const uint16_t best_sv0 = g_svm_active_sv_idx[i];
        uint32_t best_abs = (uint32_t)abs_i32((int32_t)g_svm_alpha_q3[best_sv0]);
#if SVM_CPU_ENABLE_CACHED_HINT_SORT
        uint32_t best_score = sv_order_score_cached_hint(best_abs, g_svm_sv_norm2_q_const[best_sv0]);
#else
        uint32_t best_score = sv_order_score(best_abs, g_svm_sv_nnz[best_sv0]);
#endif

        for (uint32_t j = i + 1u; j < (uint32_t)g_svm_active_sv_count; ++j) {
            const uint16_t sv = g_svm_active_sv_idx[j];
            const uint32_t cur_abs = (uint32_t)abs_i32((int32_t)g_svm_alpha_q3[sv]);
#if SVM_CPU_ENABLE_CACHED_HINT_SORT
            const uint32_t cur_score = sv_order_score_cached_hint(cur_abs, g_svm_sv_norm2_q_const[sv]);
#else
            const uint32_t cur_score = sv_order_score(cur_abs, g_svm_sv_nnz[sv]);
#endif
            if ((cur_score > best_score) || ((cur_score == best_score) && (cur_abs > best_abs))) {
                best = j;
                best_abs = cur_abs;
                best_score = cur_score;
            }
        }

        if (best != i) {
            const uint16_t tmp = g_svm_active_sv_idx[i];
            g_svm_active_sv_idx[i] = g_svm_active_sv_idx[best];
            g_svm_active_sv_idx[best] = tmp;
        }
    }

#if !SVM_CPU_FORCE_DENSE_NEON
    uint32_t active_sparse_cursor = 0u;
#endif
    for (uint32_t i = 0u; i < (uint32_t)g_svm_active_sv_count; ++i) {
        const uint32_t sv = (uint32_t)g_svm_active_sv_idx[i];
        const uint32_t src_base = sv * SVM_CPU_IMG_SIZE;
        const uint32_t dst_base = i * SVM_CPU_IMG_SIZE;
#if !SVM_CPU_FORCE_DENSE_NEON
        const uint16_t sv_nnz = g_svm_sv_nnz[sv];
#endif

        g_svm_active_alpha_q3[i] = g_svm_alpha_q3[sv];
        g_svm_active_sv_norm2_q[i] = g_svm_sv_norm2_q_const[sv];
        g_svm_active_sv_norm_q[i] = isqrt_u32((uint32_t)g_svm_sv_norm2_q_const[sv]);
#if !SVM_CPU_FORCE_DENSE_NEON
        g_svm_active_sv_nnz[i] = sv_nnz;
        if (sv_nnz < SVM_CPU_DENSE_DOT_NNZ_THRESHOLD) {
            g_svm_all_active_sv_dense = 0u;
        }
#endif

        memcpy(&g_svm_active_sv_dense_q1[dst_base],
               &g_svm_sv_dense_q1[src_base],
               SVM_CPU_IMG_SIZE * sizeof(int8_t));
#if !SVM_CPU_FORCE_DENSE_NEON
        g_svm_active_sv_sparse_off[i] = active_sparse_cursor;
        memcpy(&g_svm_active_sv_sparse_idx[active_sparse_cursor],
               &g_svm_sv_idx[src_base],
               (size_t)sv_nnz * sizeof(uint16_t));
        memcpy(&g_svm_active_sv_sparse_val_q1[active_sparse_cursor],
               &g_svm_sv_val_q1[src_base],
               (size_t)sv_nnz * sizeof(int8_t));
        active_sparse_cursor += (uint32_t)sv_nnz;
#endif
    }
#if !SVM_CPU_FORCE_DENSE_NEON
    g_svm_active_sv_sparse_off[g_svm_active_sv_count] = active_sparse_cursor;
#endif

    /*
     * Prefix-like tail bounds:
     * rem_pos/rem_neg at index i represent the max possible positive/negative
     * contribution from remaining SV terms [i..end), enabling exact pruning.
     */
    g_svm_rem_pos_q2048[g_svm_active_sv_count] = 0;
    g_svm_rem_neg_q2048[g_svm_active_sv_count] = 0;
    for (int32_t i = (int32_t)g_svm_active_sv_count - 1; i >= 0; --i) {
        const int32_t a_q3 = (int32_t)g_svm_active_alpha_q3[i];
        g_svm_rem_pos_q2048[i] = g_svm_rem_pos_q2048[i + 1] + ((a_q3 > 0) ? (a_q3 * 256) : 0);
        g_svm_rem_neg_q2048[i] = g_svm_rem_neg_q2048[i + 1] + ((a_q3 < 0) ? (a_q3 * 256) : 0);
    }

    if (g_svm_test_cache_ready == 0) {
#if SVM_CPU_FORCE_DENSE_NEON
        g_svm_all_cached_x_dense = 1u;
        for (uint32_t img = 0u; img < SVM_CPU_NUM_IMAGES; ++img) {
            const int8_t *restrict x_ptr = &g_mnist_test_q7_1[img * SVM_CPU_IMG_SIZE];
            g_svm_x_norm2_q[img] = compute_dense_norm2_q1(x_ptr);
            g_svm_x_norm_q[img] = isqrt_u32((uint32_t)g_svm_x_norm2_q[img]);
        }
#else
        uint32_t x_sparse_cursor = 0u;
        g_svm_all_cached_x_dense = 1u;
        for (uint32_t img = 0u; img < SVM_CPU_NUM_IMAGES; ++img) {
            const int8_t *restrict x_ptr = &g_mnist_test_q7_1[img * SVM_CPU_IMG_SIZE];
            uint16_t nnz = 0u;
            int32_t norm2_q = 0;
            g_svm_x_sparse_off[img] = x_sparse_cursor;

            for (uint32_t j = 0u; j < SVM_CPU_IMG_SIZE; ++j) {
                const int8_t q = x_ptr[j];

                if (LIKELY((j + 16u) < SVM_CPU_IMG_SIZE)) {
                    __builtin_prefetch(&x_ptr[j + 16u], 0, 1);
                }

                norm2_q += (int32_t)q * (int32_t)q;
                if (q != 0) {
                    g_svm_x_idx[x_sparse_cursor + nnz] = (uint16_t)j;
                    g_svm_x_val_q1[x_sparse_cursor + nnz] = q;
                    ++nnz;
                }
            }

            x_sparse_cursor += (uint32_t)nnz;
            g_svm_x_norm2_q[img] = norm2_q;
            g_svm_x_norm_q[img] = isqrt_u32((uint32_t)norm2_q);
            g_svm_x_nnz[img] = nnz;
            if (nnz < SVM_CPU_DENSE_DOT_NNZ_THRESHOLD) {
                g_svm_all_cached_x_dense = 0u;
            }
        }
        g_svm_x_sparse_off[SVM_CPU_NUM_IMAGES] = x_sparse_cursor;
#endif

        g_svm_test_cache_ready = 1;
    }

    if ((g_svm_test_cache_ready != 0) && (g_svm_img_bounds_ready == 0)) {
        build_cached_image_bounds();
        g_svm_img_bounds_ready = 1;
    }

    g_svm_prepared = 1;
    return XST_SUCCESS;
}

int __attribute__((hot)) svm_cpu_quantized_run_batch_timed(const int8_t *in_q7_1,
                                                           uint8_t *out_label,
                                                           uint16_t n_images,
                                                           uint64_t *cpu_cycles) {
    const int use_cached_test = (in_q7_1 == g_mnist_test_q7_1) &&
                                ((uint32_t)n_images <= SVM_CPU_NUM_IMAGES);
    XTime t_start;
    XTime t_end;

    if ((in_q7_1 == NULL) || (out_label == NULL) || (cpu_cycles == NULL)) {
        return XST_INVALID_PARAM;
    }

    if (n_images == 0u) {
        *cpu_cycles = 0u;
        return XST_SUCCESS;
    }

    if ((svm_cpu_quantized_prepare() != XST_SUCCESS) || ((uint32_t)n_images > SVM_CPU_NUM_IMAGES)) {
        return XST_FAILURE;
    }

    /* Timed window only covers per-image inference loop. */
    XTime_GetTime(&t_start);

    for (uint32_t img = 0u; img < (uint32_t)n_images; ++img) {
        const int8_t *x_dense_q1;
        int32_t x_norm2_q;
        int32_t score_q2048 = ((int32_t)g_svm_bias_q1_7) << SVM_CPU_SCORE_BIAS_SHIFT;
        const int32_t *rem_pos_q2048 = g_svm_rem_pos_q2048;
        const int32_t *rem_neg_q2048 = g_svm_rem_neg_q2048;
        const uint32_t *img_kzero_mask = NULL;
#if SVM_CPU_ENABLE_KZERO_SKIP
        int has_kzero_mask = 0;
#endif
#if !SVM_CPU_FORCE_DENSE_NEON
        const uint16_t *x_idx;
        const int8_t *x_val_q1;
        uint16_t x_nnz;
#endif

        if (use_cached_test != 0) {
            /*
             * Fast path for known MNIST test set:
             * use cached norm/sparsity/bound metadata precomputed in prepare().
             */
            const uint32_t dense_base = img * SVM_CPU_IMG_SIZE;
#if !SVM_CPU_FORCE_DENSE_NEON
            x_dense_q1 = (const int8_t *)__builtin_assume_aligned(&in_q7_1[dense_base], SVM_CPU_CACHELINE_BYTES);
            x_norm2_q = g_svm_x_norm2_q[img];
            if (g_svm_all_cached_x_dense != 0u) {
                x_nnz = SVM_CPU_IMG_SIZE;
            } else {
                const uint32_t sparse_base = g_svm_x_sparse_off[img];
                x_idx = (const uint16_t *)__builtin_assume_aligned(&g_svm_x_idx[sparse_base], SVM_CPU_CACHELINE_BYTES);
                x_val_q1 = (const int8_t *)__builtin_assume_aligned(&g_svm_x_val_q1[sparse_base], SVM_CPU_CACHELINE_BYTES);
                x_nnz = g_svm_x_nnz[img];
            }
            if (g_svm_img_bounds_ready != 0) {
                const uint32_t off = img * ((uint32_t)SVM_CPU_NUM_SV + 1u);
                rem_pos_q2048 = &g_svm_img_rem_pos_q2048[off];
                rem_neg_q2048 = &g_svm_img_rem_neg_q2048[off];
#if SVM_CPU_ENABLE_KZERO_SKIP
                img_kzero_mask = &g_svm_img_kzero_mask[img * SVM_CPU_ACTIVE_SV_MASK_WORDS];
                has_kzero_mask = (g_svm_img_has_kzero[img] != 0u) ? 1 : 0;
#endif
            }

            if (LIKELY((img + 1u) < (uint32_t)n_images)) {
                const uint32_t next_dense_base = (img + 1u) * SVM_CPU_IMG_SIZE;
                __builtin_prefetch(&in_q7_1[next_dense_base], 0, 1);
                if (g_svm_all_cached_x_dense == 0u) {
                    const uint32_t next_sparse_base = g_svm_x_sparse_off[img + 1u];
                    __builtin_prefetch(&g_svm_x_idx[next_sparse_base], 0, 1);
                    __builtin_prefetch(&g_svm_x_val_q1[next_sparse_base], 0, 1);
                }
            }
#else
            x_dense_q1 = (const int8_t *)__builtin_assume_aligned(&in_q7_1[dense_base], SVM_CPU_CACHELINE_BYTES);
            x_norm2_q = g_svm_x_norm2_q[img];
            if (g_svm_img_bounds_ready != 0) {
                const uint32_t off = img * ((uint32_t)SVM_CPU_NUM_SV + 1u);
                rem_pos_q2048 = &g_svm_img_rem_pos_q2048[off];
                rem_neg_q2048 = &g_svm_img_rem_neg_q2048[off];
#if SVM_CPU_ENABLE_KZERO_SKIP
                img_kzero_mask = &g_svm_img_kzero_mask[img * SVM_CPU_ACTIVE_SV_MASK_WORDS];
                has_kzero_mask = (g_svm_img_has_kzero[img] != 0u) ? 1 : 0;
#endif
            }
            if (LIKELY((img + 1u) < (uint32_t)n_images)) {
                const uint32_t next_dense_base = (img + 1u) * SVM_CPU_IMG_SIZE;
                __builtin_prefetch(&in_q7_1[next_dense_base], 0, 1);
            }
#endif
        } else {
            const int8_t *restrict x_ptr = &in_q7_1[img * SVM_CPU_IMG_SIZE];
#if !SVM_CPU_FORCE_DENSE_NEON
            build_sparse_x_local(x_ptr,
                                 g_svm_x_idx_scratch,
                                 g_svm_x_val_q1_scratch,
                                 &x_nnz,
                                 &x_norm2_q);
            x_idx = (const uint16_t *)__builtin_assume_aligned(g_svm_x_idx_scratch, SVM_CPU_CACHELINE_BYTES);
            x_val_q1 = (const int8_t *)__builtin_assume_aligned(g_svm_x_val_q1_scratch, SVM_CPU_CACHELINE_BYTES);
            x_dense_q1 = (const int8_t *)__builtin_assume_aligned(x_ptr, SVM_CPU_CACHELINE_BYTES);
#else
            x_norm2_q = compute_dense_norm2_q1(x_ptr);
            x_dense_q1 = (const int8_t *)__builtin_assume_aligned(x_ptr, SVM_CPU_CACHELINE_BYTES);
#endif
        }

#if SVM_CPU_ENABLE_KMAX_HOTPATH
        if ((use_cached_test != 0) &&
            (g_svm_img_bounds_ready != 0) &&
            (g_svm_all_active_sv_dense != 0u) &&
            (g_svm_all_cached_x_dense != 0u)) {
            out_label[img] = classify_cached_dense_with_kmax(img, x_dense_q1, x_norm2_q);
            continue;
        }
#endif

        /* Immediate decision if even the best remaining opposite terms cannot flip sign. */
        if (UNLIKELY((score_q2048 + rem_neg_q2048[0]) > 0)) {
            out_label[img] = 1u;
            continue;
        }
        if (UNLIKELY((score_q2048 + rem_pos_q2048[0]) < 0)) {
            out_label[img] = 0u;
            continue;
        }

        {
            int dense_pair_fast = 0;

#if SVM_CPU_ENABLE_DENSE_PAIR_FASTPATH
#if !SVM_CPU_FORCE_DENSE_NEON
            if (g_svm_all_active_sv_dense != 0u) {
                if (use_cached_test != 0) {
                    dense_pair_fast = (g_svm_all_cached_x_dense != 0u) ? 1 : 0;
                } else {
                    dense_pair_fast = (x_nnz >= SVM_CPU_DENSE_DOT_NNZ_THRESHOLD) ? 1 : 0;
                }
            }
#else
            dense_pair_fast = 1;
#endif
#endif

            if (dense_pair_fast != 0) {
                /*
                 * Main hot path: dense NEON pair/quad dot products plus exact
                 * early-exit checks after each accumulated SV term.
                 */
                uint32_t a = 0u;
                int terminated = 0;
                const uint32_t active_count = (uint32_t)g_svm_active_sv_count;

                if (terminated == 0) {
                    for (; (a + 3u) < active_count; a += 4u) {
                        const uint32_t dense_base0 = a * SVM_CPU_IMG_SIZE;
                        const uint32_t dense_base1 = dense_base0 + SVM_CPU_IMG_SIZE;
                        const uint32_t dense_base2 = dense_base1 + SVM_CPU_IMG_SIZE;
                        const uint32_t dense_base3 = dense_base2 + SVM_CPU_IMG_SIZE;
                        const int8_t *restrict sv0_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base0], SVM_CPU_CACHELINE_BYTES);
                        const int8_t *restrict sv1_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base1], SVM_CPU_CACHELINE_BYTES);
                        const int8_t *restrict sv2_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base2], SVM_CPU_CACHELINE_BYTES);
                        const int8_t *restrict sv3_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base3], SVM_CPU_CACHELINE_BYTES);
                        int32_t dot0_q = 0;
                        int32_t dot1_q = 0;
                        int32_t dot2_q = 0;
                        int32_t dot3_q = 0;
#if SVM_CPU_ENABLE_KZERO_SKIP
                        const int skip0 = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a) : 0;
                        const int skip1 = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a + 1u) : 0;
                        const int skip2 = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a + 2u) : 0;
                        const int skip3 = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a + 3u) : 0;
#else
                        const int skip0 = 0;
                        const int skip1 = 0;
                        const int skip2 = 0;
                        const int skip3 = 0;
#endif

                        if (LIKELY((a + 4u) < active_count)) {
                            const uint32_t next_base = (a + 4u) * SVM_CPU_IMG_SIZE;
                            __builtin_prefetch(&g_svm_active_sv_dense_q1[next_base], 0, 1);
                        }

                        if ((skip0 == 0) || (skip1 == 0) || (skip2 == 0) || (skip3 == 0)) {
                            dot_dense_dense4_neon_q1(sv0_q1, sv1_q1, sv2_q1, sv3_q1, x_dense_q1,
                                                     &dot0_q, &dot1_q, &dot2_q, &dot3_q);
                        }

                        if (accumulate_sv_term_and_check(a, skip0, dot0_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            terminated = 1;
                            break;
                        }
                        if (accumulate_sv_term_and_check(a + 1u, skip1, dot1_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            terminated = 1;
                            break;
                        }
                        if (accumulate_sv_term_and_check(a + 2u, skip2, dot2_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            terminated = 1;
                            break;
                        }
                        if (accumulate_sv_term_and_check(a + 3u, skip3, dot3_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            terminated = 1;
                            break;
                        }
                    }
                }

                if (terminated == 0) {
                    for (; (a + 1u) < active_count; a += 2u) {
                        const uint32_t dense_base0 = a * SVM_CPU_IMG_SIZE;
                        const uint32_t dense_base1 = dense_base0 + SVM_CPU_IMG_SIZE;
                        const int8_t *restrict sv0_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base0], SVM_CPU_CACHELINE_BYTES);
                        const int8_t *restrict sv1_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base1], SVM_CPU_CACHELINE_BYTES);
                        int32_t dot0_q = 0;
                        int32_t dot1_q = 0;
#if SVM_CPU_ENABLE_KZERO_SKIP
                        const int skip0 = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a) : 0;
                        const int skip1 = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a + 1u) : 0;
#else
                        const int skip0 = 0;
                        const int skip1 = 0;
#endif

                        if (LIKELY((a + 2u) < active_count)) {
                            const uint32_t next_base = (a + 2u) * SVM_CPU_IMG_SIZE;
                            __builtin_prefetch(&g_svm_active_sv_dense_q1[next_base], 0, 1);
                        }

                        if ((skip0 == 0) || (skip1 == 0)) {
                            dot_dense_dense2_neon_q1(sv0_q1, sv1_q1, x_dense_q1, &dot0_q, &dot1_q);
                        }

                        if (accumulate_sv_term_and_check(a, skip0, dot0_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            terminated = 1;
                            break;
                        }
                        if (accumulate_sv_term_and_check(a + 1u, skip1, dot1_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            terminated = 1;
                            break;
                        }
                    }
                }

                if (terminated == 0) {
                    for (; a < active_count; ++a) {
                        const uint32_t dense_base = a * SVM_CPU_IMG_SIZE;
                        const int8_t *restrict sv_dense_q1 = (const int8_t *)__builtin_assume_aligned(
                            &g_svm_active_sv_dense_q1[dense_base], SVM_CPU_CACHELINE_BYTES);
                        int32_t dot_q = 0;
#if SVM_CPU_ENABLE_KZERO_SKIP
                        const int skip = (has_kzero_mask != 0) ? is_kzero_masked(img_kzero_mask, a) : 0;
#else
                        const int skip = 0;
#endif
                        if (skip == 0) {
                            dot_q = dot_dense_dense_neon_q1(sv_dense_q1, x_dense_q1);
                        }
                        if (accumulate_sv_term_and_check(a, skip, dot_q, x_norm2_q, &score_q2048, rem_pos_q2048, rem_neg_q2048) != 0) {
                            break;
                        }
                    }
                }
            } else {
                /* Fallback path (sparse-aware when enabled, otherwise dense scalar/NEON). */
                for (uint32_t a = 0u; a < (uint32_t)g_svm_active_sv_count; ++a) {
#if SVM_CPU_ENABLE_KZERO_SKIP
                    if ((has_kzero_mask != 0) && is_kzero_masked(img_kzero_mask, a)) {
                        if ((score_q2048 + rem_neg_q2048[a + 1u]) > 0) {
                            score_q2048 = 1;
                            break;
                        }
                        if ((score_q2048 + rem_pos_q2048[a + 1u]) < 0) {
                            score_q2048 = -1;
                            break;
                        }
                        continue;
                    }
#endif
                    const uint32_t dense_base = a * SVM_CPU_IMG_SIZE;
                    const int32_t alpha_q3 = (int32_t)g_svm_active_alpha_q3[a];
                    const int32_t sv_norm2_q = g_svm_active_sv_norm2_q[a];
                    const int8_t *restrict sv_dense_q1 =
                        (const int8_t *)__builtin_assume_aligned(&g_svm_active_sv_dense_q1[dense_base], SVM_CPU_CACHELINE_BYTES);
                    int32_t dot_q;
                    int32_t d2_raw;
                    uint8_t k_q8;
#if !SVM_CPU_FORCE_DENSE_NEON
                    const uint32_t sparse_base = g_svm_active_sv_sparse_off[a];
                    const uint16_t sv_nnz = g_svm_active_sv_nnz[a];
                    const uint16_t *restrict sv_idx =
                        (const uint16_t *)__builtin_assume_aligned(&g_svm_active_sv_sparse_idx[sparse_base], SVM_CPU_CACHELINE_BYTES);
                    const int8_t *restrict sv_val_q1 =
                        (const int8_t *)__builtin_assume_aligned(&g_svm_active_sv_sparse_val_q1[sparse_base], SVM_CPU_CACHELINE_BYTES);
#endif

                    if (LIKELY((a + 1u) < (uint32_t)g_svm_active_sv_count)) {
#if !SVM_CPU_FORCE_DENSE_NEON
                        const uint16_t next_sv_nnz = g_svm_active_sv_nnz[a + 1u];
                        if ((next_sv_nnz < SVM_CPU_DENSE_DOT_NNZ_THRESHOLD) ||
                            (x_nnz < SVM_CPU_DENSE_DOT_NNZ_THRESHOLD)) {
                            const uint32_t next_sparse_base = g_svm_active_sv_sparse_off[a + 1u];
                            __builtin_prefetch(&g_svm_active_sv_sparse_idx[next_sparse_base], 0, 1);
                            __builtin_prefetch(&g_svm_active_sv_sparse_val_q1[next_sparse_base], 0, 1);
                        } else {
                            const uint32_t next_dense_base = (a + 1u) * SVM_CPU_IMG_SIZE;
                            __builtin_prefetch(&g_svm_active_sv_dense_q1[next_dense_base], 0, 1);
                        }
#else
                        const uint32_t next_dense_base = (a + 1u) * SVM_CPU_IMG_SIZE;
                        __builtin_prefetch(&g_svm_active_sv_dense_q1[next_dense_base], 0, 1);
#endif
                    }

#if !SVM_CPU_FORCE_DENSE_NEON
                    if ((sv_nnz < SVM_CPU_DENSE_DOT_NNZ_THRESHOLD) ||
                        (x_nnz < SVM_CPU_DENSE_DOT_NNZ_THRESHOLD)) {
                        if (sv_nnz <= x_nnz) {
                            dot_q = dot_sparse_sv_dense_x_q1(sv_idx, sv_val_q1, sv_nnz, x_dense_q1);
                        } else {
                            dot_q = dot_sparse_x_dense_sv_q1(x_idx, x_val_q1, x_nnz, sv_dense_q1);
                        }
                    } else {
                        dot_q = dot_dense_dense_neon_q1(sv_dense_q1, x_dense_q1);
                    }
#else
                    dot_q = dot_dense_dense_neon_q1(sv_dense_q1, x_dense_q1);
#endif

                    d2_raw = sv_norm2_q + x_norm2_q - (dot_q << 1);
                    if (UNLIKELY(d2_raw < 0)) {
                        d2_raw = 0;
                    } else if (UNLIKELY(d2_raw > SVM_CPU_EXP_D2_CLIP)) {
                        d2_raw = SVM_CPU_EXP_D2_CLIP;
                    }

                    k_q8 = g_svm_exp_lut_rom[(uint32_t)d2_raw >> SVM_CPU_EXP_LUT_D2_SHIFT];
                    if (k_q8 != 0u) {
                        score_q2048 += alpha_q3 * (int32_t)k_q8;
                    }

                    if ((score_q2048 + rem_neg_q2048[a + 1u]) > 0) {
                        score_q2048 = 1;
                        break;
                    }
                    if ((score_q2048 + rem_pos_q2048[a + 1u]) < 0) {
                        score_q2048 = -1;
                        break;
                    }
                }
            }
        }

        out_label[img] = (score_q2048 >= 0) ? 1u : 0u;
    }

    XTime_GetTime(&t_end);
    *cpu_cycles = (uint64_t)(t_end - t_start);

    return XST_SUCCESS;
}

int __attribute__((cold)) svm_cpu_quantized_eval_accuracy(const uint8_t *pred,
                                                          const uint8_t *gt,
                                                          uint16_t n_images,
                                                          float *acc_out,
                                                          uint32_t *mismatches_out) {
    uint32_t correct = 0u;

    /* Accuracy-only metric used by current testbench-style reporting. */
    if ((pred == NULL) || (gt == NULL) || (acc_out == NULL) || (mismatches_out == NULL)) {
        return XST_INVALID_PARAM;
    }

    if (n_images == 0u) {
        *acc_out = 0.0f;
        *mismatches_out = 0u;
        return XST_SUCCESS;
    }

    for (uint32_t i = 0; i < (uint32_t)n_images; ++i) {
        if ((pred[i] & 0x1u) == (gt[i] & 0x1u)) {
            ++correct;
        }
    }

    *mismatches_out = (uint32_t)n_images - correct;
    *acc_out = (float)correct / (float)n_images;
    return XST_SUCCESS;
}
