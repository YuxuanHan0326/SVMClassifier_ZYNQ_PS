# SVM Accelerator App (ZedBoard PS + PL)

## 1. Overview
- This project runs MNIST binary SVM classification on both:
  - PL accelerator IP (`svm_classifier_ip`) through AXI DMA.
  - PS software kernel (`svm_cpu_quantized_*`).
- Current default execution order in `app_component/src/main.c`:
  1. Run PL batch on 2601 images.
  2. Evaluate PL accuracy.
  3. Run PS batch on the same 2601 images.
  4. Evaluate PS accuracy.
- UART output includes timing, throughput, and accuracy (no confusion matrix).

## 2. Key Files
- `app_component/src/main.c`
  - Top-level control flow, timing conversion, accuracy print, optional PMU profiling.
- `app_component/src/svm_ps_driver.c/.h`
  - PS-side hardware driver:
    - DMA + IP init.
    - Batch launch/wait.
    - PL timing collection.
- `app_component/src/svm_cpu_quantized.c/.h`
  - PS quantized SVM implementation (Q-format + LUT + NEON hot paths).
- `app_component/src/mnist_q7_1_data.c/.h`
  - Quantized test images (Q7.1) and ground truth for PL-side evaluation.
- `app_component/src/svm_cpu_model_data.c/.h`
  - Model parameters and labels used by the PS quantized kernel.
- `app_component/src/UserConfig.cmake`
  - Compiler optimization flags.
- `app_component/src/lscript.ld`
  - Linker script (includes OCM placement used by hot buffers).

## 3. Runtime Flow (main)
- Enable caches:
  - `Xil_ICacheEnable()`
  - `Xil_DCacheEnable()`
  - `Xil_L2CacheEnable()`
- Initialize hardware:
  - `svm_init_hw()`
- PL batch run:
  - `svm_run_batch_timed(...)`
  - Prints:
    - `PL kernel_cycles/kernel_time_us`
    - `PL dma_cycles/dma_time_us/images_per_s`
    - `PL mismatches/accuracy/acc_ok`
- PS batch run:
  - `svm_cpu_quantized_run_batch_timed(...)` directly or via PMU wrapper.
  - Prints:
    - `PS_QUANTIZED cycles/time_us/images_per_s`
    - `PS_QUANTIZED mismatches/accuracy/acc_ok`
- PMU profiling is controlled by macros in `main.c`:
  - `ENABLE_PMU_PROFILING`
  - `ENABLE_PMU_MULTI_CONFIG`

## 4. Timing Definitions
- PL kernel time:
  - `XSvm_classifier_ip_Start()` to `XSvm_classifier_ip_IsDone()==1`.
- PL DMA time:
  - MM2S launch to S2MM completion.
- PS time:
  - Inside `svm_cpu_quantized_run_batch_timed()`, around the per-image inference loop.
- Time conversion:
  - `time_us = cycles * 1e6 / COUNTS_PER_SECOND`.

## 5. Data / Interface Contract (PL path)
- Input image size: 784 bytes (28x28).
- Batch size used in this app: 2601 images.
- DMA lengths:
  - TX (MM2S): `784 * n_images` bytes.
  - RX (S2MM): `n_images` bytes.
- Output parsing:
  - Final label uses bit0 (`out_label[i] &= 0x1u`).

## 6. Build Notes
- Intended build flow: Vitis embedded flow (`empyro.bat` / IDE build).
- Local syntax check example:
  - `arm-none-eabi-gcc -fsyntax-only ... main.c svm_ps_driver.c svm_cpu_quantized.c`
- If git reports `dubious ownership`, add safe directory:
  - `git config --global --add safe.directory D:/SVM_Accelerator/SVM_Accelerator_Project`

## 7. Typical UART Output (simplified)
- `TB_IMAGES=2601`
- `PL kernel_cycles=... kernel_time_us=...`
- `PL dma_cycles=... dma_time_us=... images_per_s=...`
- `PL mismatches=... accuracy=... threshold=0.98 acc_ok=...`
- `PS_QUANTIZED cycles=... time_us=... images_per_s=...`
- `PS_QUANTIZED mismatches=... accuracy=... threshold=0.98 acc_ok=...`
- Optional PMU lines when profiling is enabled.

## 8. Accuracy Target
- `TB_MIN_ACCURACY` is set to `0.98`.
- `acc_ok=1` when measured accuracy is `>= 0.98`.

## 9. Notes on Async APIs
- `svm_ps_driver.c` also provides:
  - `svm_run_batch_async_start(...)`
  - `svm_run_batch_async_wait(...)`
- They are currently not used by default main flow (serial PL -> PS mode), but kept for controlled experiments.
