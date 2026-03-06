# ZedBoard SVM PL/PS Parallel Degradation Experiment

## 1. Experiment Goal
This repository is currently focused on one question:

**When PL (`svm_classifier_ip + AXI DMA`) and PS (ARM-side quantized SVM) run in parallel, does PL suffer significant performance degradation due to memory arbitration?**

The code path for this experiment is in:
- `app_component/src/main.c` (`ENABLE_CONCURRENCY_PROOF_TEST=1`)
- `app_component/src/svm_ps_driver.c`
- `app_component/src/svm_cpu_quantized.c`

## 2. Test Setup
- Board: ZedBoard (Zynq-7020, Cortex-A9 + PL accelerator)
- Dataset size: `n_pl=2601`, `n_ps=2601`
- Repetitions: `30` runs
- Warmup discarded: first `3` runs
- Polling frequency for PL async completion latch during PS run: `hook_period=1` (poll every PS image)

Experiment modes per run:
1. `PL_ONLY`: PL baseline latency
2. `PS_ONLY`: PS baseline latency
3. `PL + CPU_SPIN`: PL with compute-only PS load (low memory traffic)
4. `PL + CPU_MEM_STRESS`: PL with PS memory-stream pressure
5. `PL + PS_SVM`: real PL+PS parallel inference

## 3. How To Run
1. Build app:
```bat
cmd.exe /C "set CC= && set CXX= && empyro.bat build_app -s d:\SVM_Accelerator\SVM_Accelerator_Project\app_component\src -b d:\SVM_Accelerator\SVM_Accelerator_Project\app_component\build"
```
2. Program bitstream/ELF in Vitis and launch on hardware.
3. Open UART console and capture lines:
   - `PROOF MEDIAN ...`
   - `PROOF RATIOS ...`
   - `PROOF CLAIM ...`

## 4. Measured Results (Latest Run)
From UART output:

- `PROOF MEDIAN pl_only_us=5371 ps_only_us=224790 spin_pl_us=5376 mem_pl_us=5392 par_total_us=228820 par_pl_us=5429 par_ps_us=225643`
- `PROOF RATIOS spin_slowdown=1.001 mem_slowdown=1.004 pl_slowdown=1.011 ps_slowdown=1.004 serial_over_parallel=1.006`
- `PROOF CLAIM memory_arbiter=0`
- Accuracy:
  - `PL`: `12` mismatches, `0.995386`
  - `PS`: `8` mismatches, `0.996924`

## 5. Analysis & Conclusion
Key observations:
1. `PL_ONLY -> par_pl` changed from `5371 us` to `5429 us` (`+58 us`, about `+1.1%`).
2. `spin_pl` and `mem_pl` are both close to `pl_only` (`1.001x` and `1.004x`), showing no large PL slowdown under added PS load.
3. Real parallel run also shows only small PL impact (`pl_slowdown=1.011`), not a significant degradation.
4. Overall completion time (`par_total`) is still dominated by PS runtime scale (about 225 ms vs PL about 5.4 ms).

Conclusion:
- **Under this workload and configuration, PL does not exhibit significant performance degradation when running in parallel with PS.**
- **The experiment does not support a severe memory-arbitration bottleneck on PL claim.**
- Current result is consistent with mild overhead only, not major contention.

## 6. Notes
- `PROOF CLAIM memory_arbiter` is an automated heuristic based on relative slowdown between control modes.
- If future testing needs to amplify contention effects, increase sustained PL workload (for example repeated back-to-back PL batches) and retest with the same method.