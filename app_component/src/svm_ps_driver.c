#include "svm_ps_driver.h"

#include "xaxidma.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xsvm_classifier_ip.h"

#ifndef XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#ifndef XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ
#define XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ XPAR_CPU_CORE_CLOCK_FREQ_HZ
#endif

#include "xtime_l.h"

#define SVM_IMG_SIZE_BYTES 784u
#define SVM_DMA_POLL_TIMEOUT 200000000u

/* Single static driver instances used by the app. */
static XAxiDma g_dma;
static XSvm_classifier_ip g_svm_ip;
static int g_is_initialized = 0;

/*
 * Async batch context:
 * Stored between async_start() and async_wait() so wait side can validate
 * the expected buffer/batch and compute cycle deltas.
 */
static int g_batch_in_flight = 0;
static uint8_t *g_async_out_label = NULL;
static uint16_t g_async_n_images = 0u;
static uint32_t g_async_rx_len = 0u;
static XTime g_async_t_dma_start = 0;
static XTime g_async_t_kernel_start = 0;

int svm_init_hw(void) {
    int status;
    XAxiDma_Config *dma_cfg;

    dma_cfg = XAxiDma_LookupConfig(XPAR_XAXIDMA_0_BASEADDR);
    if (dma_cfg == NULL) {
        return XST_DEVICE_NOT_FOUND;
    }

    status = XAxiDma_CfgInitialize(&g_dma, dma_cfg);
    if (status != XST_SUCCESS) {
        return status;
    }

    if (XAxiDma_HasSg(&g_dma) != 0) {
        return XST_FAILURE;
    }

    status = XSvm_classifier_ip_Initialize(&g_svm_ip, XPAR_XSVM_CLASSIFIER_IP_0_BASEADDR);
    if (status != XST_SUCCESS) {
        return status;
    }

    XSvm_classifier_ip_DisableAutoRestart(&g_svm_ip);
    g_is_initialized = 1;
    return XST_SUCCESS;
}

int svm_run_batch_async_start(const int8_t *in_q7_1,
                              uint8_t *out_label,
                              uint16_t n_images) {
    int status;
    uint32_t tx_len;
    uint32_t rx_len;

    if ((in_q7_1 == NULL) || (out_label == NULL)) {
        return XST_INVALID_PARAM;
    }

    if (!g_is_initialized) {
        status = svm_init_hw();
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    if (g_batch_in_flight != 0) {
        return XST_DEVICE_BUSY;
    }

    if (n_images == 0u) {
        return XST_SUCCESS;
    }

    /* Contract: one image = 784 input bytes, one output byte. */
    tx_len = (uint32_t)n_images * SVM_IMG_SIZE_BYTES;
    rx_len = (uint32_t)n_images;

    if (tx_len > g_dma.TxBdRing.MaxTransferLen) {
        return XST_INVALID_PARAM;
    }
    if (rx_len > g_dma.RxBdRing[0].MaxTransferLen) {
        return XST_INVALID_PARAM;
    }

    Xil_DCacheFlushRange((INTPTR)in_q7_1, tx_len);
    Xil_DCacheInvalidateRange((INTPTR)out_label, rx_len);

    /* Program control register before launching stream traffic. */
    XSvm_classifier_ip_Set_n_images(&g_svm_ip, n_images);

    /* Launch RX path first to avoid losing early output beats. */
    status = XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)out_label, rx_len, XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) {
        return status;
    }

    XTime_GetTime(&g_async_t_dma_start);
    status = XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)in_q7_1, tx_len, XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) {
        return status;
    }

    XSvm_classifier_ip_Start(&g_svm_ip);
    XTime_GetTime(&g_async_t_kernel_start);

    g_batch_in_flight = 1;
    g_async_out_label = out_label;
    g_async_n_images = n_images;
    g_async_rx_len = rx_len;
    return XST_SUCCESS;
}

int svm_run_batch_async_wait(uint8_t *out_label,
                             uint16_t n_images,
                             uint64_t *mm2s_to_s2mm_cycles,
                             uint64_t *kernel_apstart_to_done_cycles) {
    uint32_t timeout_cycles;
    int mm2s_done;
    int s2mm_done;
    int ip_done;
    XTime t_dma_end;
    XTime t_kernel_end;

    if ((out_label == NULL) || (mm2s_to_s2mm_cycles == NULL) || (kernel_apstart_to_done_cycles == NULL)) {
        return XST_INVALID_PARAM;
    }

    if (n_images == 0u) {
        *mm2s_to_s2mm_cycles = 0u;
        *kernel_apstart_to_done_cycles = 0u;
        return XST_SUCCESS;
    }

    if (g_batch_in_flight == 0) {
        return XST_FAILURE;
    }
    if ((out_label != g_async_out_label) || (n_images != g_async_n_images)) {
        return XST_INVALID_PARAM;
    }

    mm2s_done = 0;
    s2mm_done = 0;
    ip_done = 0;
    timeout_cycles = SVM_DMA_POLL_TIMEOUT;

    /*
     * Poll all three completion conditions:
     * - MM2S drain
     * - S2MM receive complete
     * - IP ap_done
     * We record individual end timestamps when each event first happens.
     */
    while (timeout_cycles > 0u) {
        if (!mm2s_done && (XAxiDma_Busy(&g_dma, XAXIDMA_DMA_TO_DEVICE) == 0u)) {
            mm2s_done = 1;
        }

        if (!s2mm_done && (XAxiDma_Busy(&g_dma, XAXIDMA_DEVICE_TO_DMA) == 0u)) {
            s2mm_done = 1;
            XTime_GetTime(&t_dma_end);
        }

        if (!ip_done && (XSvm_classifier_ip_IsDone(&g_svm_ip) != 0u)) {
            ip_done = 1;
            XTime_GetTime(&t_kernel_end);
        }

        if (mm2s_done && s2mm_done && ip_done) {
            break;
        }
        timeout_cycles--;
    }

    if (!(mm2s_done && s2mm_done && ip_done)) {
        g_batch_in_flight = 0;
        return XST_FAILURE;
    }

    *mm2s_to_s2mm_cycles = (uint64_t)(t_dma_end - g_async_t_dma_start);
    *kernel_apstart_to_done_cycles = (uint64_t)(t_kernel_end - g_async_t_kernel_start);

    /* Make fresh DMA output visible to CPU and normalize to 0/1 label bit. */
    Xil_DCacheInvalidateRange((INTPTR)out_label, g_async_rx_len);
    for (uint32_t i = 0; i < g_async_rx_len; ++i) {
        out_label[i] &= 0x1u;
    }

    g_batch_in_flight = 0;
    g_async_out_label = NULL;
    g_async_n_images = 0u;
    g_async_rx_len = 0u;
    return XST_SUCCESS;
}

int svm_run_batch_timed(const int8_t *in_q7_1,
                        uint8_t *out_label,
                        uint16_t n_images,
                        uint64_t *mm2s_to_s2mm_cycles,
                        uint64_t *kernel_apstart_to_done_cycles) {
    int status;

    if ((in_q7_1 == NULL) || (out_label == NULL) ||
        (mm2s_to_s2mm_cycles == NULL) || (kernel_apstart_to_done_cycles == NULL)) {
        return XST_INVALID_PARAM;
    }

    *mm2s_to_s2mm_cycles = 0u;
    *kernel_apstart_to_done_cycles = 0u;

    status = svm_run_batch_async_start(in_q7_1, out_label, n_images);
    if (status != XST_SUCCESS) {
        return status;
    }
    return svm_run_batch_async_wait(out_label, n_images, mm2s_to_s2mm_cycles, kernel_apstart_to_done_cycles);
}

int svm_eval_accuracy_only(const uint8_t *pred,
                           const uint8_t *gt,
                           uint16_t n_images,
                           float *acc_out,
                           uint32_t *mismatches_out) {
    uint32_t correct = 0u;

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
            correct++;
        }
    }

    *mismatches_out = (uint32_t)n_images - correct;
    *acc_out = (float)correct / (float)n_images;
    return XST_SUCCESS;
}
