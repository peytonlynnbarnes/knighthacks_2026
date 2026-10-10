# Neural-network inference acceleration and learned-policy workloads (RTX 2060 / WSL2, 36 h hackathon)

Scope: (1) MediaPipe hand models on GPU (ONNX/TensorRT), (2) fused pre/post-processing kernels, (3) LeRobot learned policies (ACT, Diffusion Policy, SmolVLA, pi0) trained on hand-teleop demos, (4) Triton / CuPy / Numba on Turing and CUDA-on-WSL2 limits. Hardware context: RTX 2060 6 GB (sm_75), WSL2 Ubuntu, LeRobot 0.3.3.

Research budget note: about 20 tool calls. Several numbers below are vendor or author-reported, and a few come from community model cards. No source gave RTX 2060-specific numbers for any of these workloads, so every 2060 figure in Inferences is an extrapolation.

---

## Q1. Which candidate gives the clearest before/after benchmark story with real user impact?

### Takeaway
The clearest story with real user impact is **diffusion-policy action-chunk latency**. LeRobot's Diffusion Policy is slow out of the box: 370 ms per call on an RTX 4090 at 10 denoising steps in fp32, and the LeRobot 0.3.3 default is 100 DDPM steps. Fewer steps, fp16, CUDA graphs, torch.compile or a fused denoise-loop kernel would each cut that by a measurable multiple, and the robot visibly moves faster. The catch is the training and data risk (see Q3). The MediaPipe-on-GPU and fused-preprocessing routes are cheaper to build but have a weak user-visible payoff, because MediaPipe hand tracking already takes roughly 10–20 ms per frame and palm detection rarely runs in video mode.

### Cited Findings
**Policy inference cost (LeRobot paper, ICLR 2026, Table 3: fp32, average of 100 forward passes, diffusion and flow models at 10 denoising steps):**
- ACT (52M params): 182.3 ms on M1 CPU, 42.7 ms on MPS, **5.0 ms on RTX 4090**, 13.8 ms on A100 — [LeRobot paper, arXiv 2602.22818](https://arxiv.org/pdf/2602.22818)
- Diffusion Policy (263M): CPU timed out 100% of the time (5 s limit), 3453.8 ms on MPS, **369.8 ms on RTX 4090**, 613.9 ms on A100 — [LeRobot paper](https://arxiv.org/pdf/2602.22818)
- pi0 (3.5B): timed out on CPU and on MPS, 209.4 ms on 4090, 569.0 ms on A100. SmolVLA (450M): 2028 ms on CPU, 721.8 ms on MPS, **99.2 ms on 4090**, 278.8 ms on A100 — [LeRobot paper](https://arxiv.org/pdf/2602.22818)
- Peak memory in fp32: ACT 211 MB, Diffusion 1.12 GB, SmolVLA 1.75 GB, **pi0 13.32 GB on GPU**. pi0 therefore does not fit a 6 GB card in fp32 — [LeRobot paper, Table 2](https://arxiv.org/pdf/2602.22818)
- The paper says ACT reaches "∼100-200Hz on high-end GPUs such as the RTX 4090 and A100" — [LeRobot paper](https://arxiv.org/pdf/2602.22818)
- **LeRobot v0.3.3 Diffusion defaults:** `noise_scheduler_type="DDPM"`, `num_train_timesteps=100`, `num_inference_steps=None`, where None means "defaults to be the same as num_train_timesteps" (i.e. **100 denoising steps**). Also `horizon=16`, `n_action_steps=8`, `n_obs_steps=2`, `vision_backbone="resnet18"`, `down_dims=(512,1024,2048)` — [configuration_diffusion.py @ v0.3.3](https://raw.githubusercontent.com/huggingface/lerobot/v0.3.3/src/lerobot/policies/diffusion/configuration_diffusion.py)
- The original Diffusion Policy paper reports that DDIM with 100 training and 10 inference iterations gives "0.1 s latency on an Nvidia 3080 GPU" in its real-world experiments — [Chi et al., arXiv 2303.04137](https://arxiv.org/pdf/2303.04137)
- Fewer steps cost quality. Crossway Diffusion reports policy quality dropping noticeably below about 40 iterations, with some tasks degrading below 60 — [arXiv 2307.01849](https://arxiv.org/pdf/2307.01849). DADP reports that a standard diffusion policy loses about 86% of its performance under one-step DDIM — [arXiv 2602.04037](https://arxiv.org/pdf/2602.04037)
- LeRobot's async inference (computing the next chunk while the current one executes) cut SmolVLA task time on SO-100 from 13.75 s to 9.70 s per episode and doubled the cubes moved in 60 s (9 → 19), with similar success rates (78.3% sync vs 73.3% async) — [LeRobot paper, Table 5](https://arxiv.org/pdf/2602.22818)

**MediaPipe baseline (why it is a weak acceleration target):**
- HandLandmarker (full) bundle, float16: 17.12 ms on CPU and 12.27 ms on GPU (Pixel 6 averages). Inputs are 192x192 and 224x224 — [MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- In video and live-stream modes, palm detection reruns only when landmark tracking fails (hand presence below `min_hand_presence_confidence`, or box IoU below `min_tracking_confidence`). Otherwise the previous frame's box is reused — [MediaPipe guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

### Inferences
- The LeRobot paper itself shows a ~74x gap between ACT (5 ms) and Diffusion (370 ms) on a 4090, so diffusion sampling is the obviously compute-bound stage. On a 2060 (roughly 1/4–1/6 the fp32 throughput of a 4090), fp32 Diffusion at 10 steps would plausibly take on the order of 1–2 s. At the 100-step LeRobot default it would be many seconds. This is an extrapolation, not measured.
- That makes a strong benchmark story: "default LeRobot DP: X s/chunk → DDIM-10 + fp16 + CUDA graphs (+ custom fused kernel): Y ms/chunk → robot moves smoothly instead of stop-and-go." The user-visible metric is task completion time or pauses between chunks, as in LeRobot Table 5.
- Speeding up MediaPipe from ~15 ms to ~3 ms produces no visible change for a 50 Hz servo loop, where the camera at ~30 fps is the bottleneck. The story is "frees CPU / enables multi-hand / higher FPS", which is weaker.
- The ACT story is weak for kernels, since ACT is already about 5 ms on a 4090 and probably 15–30 ms on a 2060 (extrapolated). The value of ACT is the autonomy demo itself, not acceleration.

### Gaps
- No measured RTX 2060 or Turing latency for any LeRobot policy was found. The team must benchmark it themselves.
- The LeRobot paper does not say whether its "10 denoising steps" Diffusion run used DDIM or truncated DDPM, or why 10 steps takes 370 ms on a 4090. It may include ResNet-18 encoding of 2 obs steps × cameras. I did not profile this.

---

## Q2. MediaPipe models on GPU (ONNX / TensorRT) and MediaPipe's own GPU support

### Takeaway
GPU acceleration of MediaPipe's Python Tasks API on desktop is poorly documented, Linux/EGL-only and unreliable. On WSL2 it should be treated as unavailable. The practical GPU route is the community ONNX ports (PINTO0309, OpenCV Zoo), run with onnxruntime-gpu and the CUDA or TensorRT execution provider. Each model is tiny: one forum report puts it at about 1 ms, and Qualcomm's NPU figures are under 1 ms. The win is real but small in absolute terms.

### Cited Findings
- Old MediaPipe desktop install docs say Linux desktop GPU requires a GPU with EGL driver support. Other versions of the same doc say desktop GPU is "currently not supported", so the docs are inconsistent — [mediapipe install.md (old)](https://github.com/google/mediapipe/blob/66b377c825c44cbe9cf26874451b9c2f682819fe/mediapipe/docs/install.md)
- Issue #5503 reports that `delegate=BaseOptions.Delegate.GPU` does not work for a Tasks model on Ubuntu 22.04 in Docker (MediaPipe 0.10.14). No resolution was seen — [mediapipe issue #5503](https://github.com/google-ai-edge/mediapipe/issues/5503)
- Third-party bindings state that GPU is Linux-only. Mediapipe.Net 0.8.9 says "only supports GPU computation on Linux", and MediaPipeUnityPlugin says GPU mode is not supported on Windows or macOS — [Mediapipe.Net.Runtime.GPU](https://feed.nuget.org/packages/Mediapipe.Net.Runtime.GPU/0.8.9); [MediaPipeUnityPlugin](https://github.com/android-notes/MediaPipeUnityPlugin)
- Issue #5897 (Windows user wanting GPU hand inference) is about protobuf conflicts with TF-GPU. It was closed in Aug 2026 with a comment that it is "fixed in MediaPipe 1.0.0". **Version-specific, and unverified what changed** — [mediapipe issue #5897](https://github.com/google-ai-edge/mediapipe/issues/5897)
- PINTO0309/hand-gesture-recognition-using-onnx "replaces the entire MediaPipe process with ONNX". It adds multi-palm detection and a simple tracker, requires onnxruntime / onnxruntime-gpu 1.12.0, and links PINTO_model_zoo `033_Hand_Detection_and_Tracking`. No latency numbers are published, and it is unclear whether anchors, NMS or the rotated crop are in-graph — [PINTO0309 repo](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx)
- OpenCV Zoo ships ONNX conversions of the MediaPipe palm detector and handpose. Its notes say the **int8-quantized version may produce invalid results** — [OpenCV Zoo handpose README](https://huggingface.co/opencv/opencv_zoo/blob/07ad80ff0aec935e1401f56f5c68b7deb1c3400e/models/handpose_estimation_mediapipe/README.md); [palm detector README](https://huggingface.co/opencv/opencv_zoo/blob/70423c2047b10a38a07f4549b245aa362febaf06/models/palm_detection_mediapipe/README.md)
- Qualcomm AI Hub reports the MediaPipe hand detector (256x256) at 0.685 ms in ONNX FP16 and 0.54 ms in TFLite on a Galaxy S24 NPU. The hand landmark detector takes an estimated 4.77 ms on an SA8775P GPU via LiteRT — [Qualcomm MediaPipe-Hand-Detection](https://huggingface.co/qualcomm/MediaPipe-Hand-Detection/blob/8aeb78e4c0efdd736608e61a88e21511d8076d24/README.md); [AI Hub profile](https://aihub.qualcomm.com/jobs/j576ql09g)
- A Jetson Orin Nano user reported about 1 ms inference after converting to ONNX and using ONNX Runtime (precision not stated; single forum post) — [NVIDIA forum](https://forums.developer.nvidia.com/t/slow-inference-on-custom-model/331015)

### Inferences
- In TensorRT FP16 on a 2060, both models should run well under 1–2 ms each (extrapolated). The rest of the pipeline (camera decode, Python glue, pre/post-processing) then dominates, and that is where a fused kernel (Q3 below) would matter.
- INT8 is risky. OpenCV Zoo already flags invalid int8 results, and calibration costs time. FP16 is the safe choice on Turing tensor cores.
- A custom-kernel angle here is weak because TensorRT/ORT already handle the convolutions well. Kernel work belongs in the pre/post-processing around the models.

### Gaps
- I did not find a PINTO_model_zoo benchmark table, or any desktop TensorRT latency for palm or landmark models (the search did not surface it). Check the `033_Hand_Detection_and_Tracking` folder directly.
- There is no authoritative current statement on whether `Delegate.GPU` works in the Python Tasks API on Linux or WSL2 for HandLandmarker in 2026 (MediaPipe 0.10.x or 1.0).

---

## Q3. Fused preprocessing / postprocessing kernels

### Takeaway
A single CUDA kernel doing affine warp (rotated hand ROI) + bilinear resize + BGR→RGB + normalize + HWC→NCHW (fp16 out) is a classic, well-scoped student kernel, with published prior art (CV-CUDA, cvGPUSpeedup). Reported speedups are large: tens of times at kernel level versus unfused ops. But the absolute savings on a 640x480 frame → 224x224 crop are probably around 1 ms, so it is only user-visible as part of an end-to-end "whole hand pipeline on GPU" story.

### Cited Findings
- NVIDIA reports ~5x to ~50x higher end-to-end throughput for a CV-CUDA GPU pipeline (downscale, normalize, reformat + ResNet-101 segmentation) versus CPU OpenCV pre/post-processing, with GPU inference in both cases. The 50x figure is on 4× L4 GPUs, and the gain depends on the pipeline and hardware — [NVIDIA blog, CV-CUDA](https://developer.nvidia.com/blog/increasing-throughput-and-reducing-costs-for-computer-vision-with-cv-cuda/)
- cvGPUSpeedup (an open-source fused-kernel library, not NVIDIA) reports about 38x kernel-execution speedup for a fused resize → convert → normalize → split sequence versus the equivalent chain of OpenCV-CUDA calls — [cvGPUSpeedup GitHub](https://github.com/morousg/cvGPUSpeedup)
- MediaPipe palm detection in video mode runs only on tracking loss, so the per-frame hot path is the landmark model's 224x224 rotated crop plus its preprocessing — [MediaPipe guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

### Inferences
- **What the kernel looks like:** one thread per output pixel (224×224×3). Compute the inverse affine transform (rotation + scale + translation from the previous frame's landmarks), bilinear-sample the BGR uint8 source (in a texture or plain global memory), swap channels, apply `(x/255 - mean)/std` or MediaPipe's [0,1] or [-1,1] scaling, and write fp16 NCHW (or NHWC, depending on what the ONNX port expects) directly into the TensorRT input binding. Postprocessing: decode 2016 SSD anchors for the 192x192 palm model (one thread per anchor), then sigmoid + threshold + weighted NMS. This is a small, data-dependent reduction for which CPU is often fine.
- The baseline OpenCV CPU chain (`warpAffine` + `cvtColor` + `astype` + normalize + `transpose`) on a 224x224 output likely takes about 0.5–2 ms in Python/NumPy, so a fused GPU kernel saves about 1 ms. That is a plausible 10–50x micro-benchmark, but negligible against a 33 ms camera frame. Extrapolated, not measured.
- **Biggest real win:** avoid host↔device copies. Upload the frame once with pinned memory (limited on WSL2, see Q4), keep it on the GPU through preprocessing → TRT → landmark decode, and copy back only 21×3 floats.
- Alternatives that need no custom kernel: CV-CUDA, NVIDIA DALI or `torch.nn.functional.grid_sample` + `affine_grid` would do the same thing in a few lines. The custom kernel is justified mainly as a learning and demo artifact.

### Gaps
- I found no published OpenCV-CPU vs fused-GPU numbers for this exact small-crop workload, and nothing on RTX 20-series.
- CV-CUDA and DALI support for WSL2 and sm_75 was not verified in this pass.

---

## Q4. Learned policies via LeRobot (ACT / Diffusion / SmolVLA / pi0): is "hand teleop → demos → train ACT → autonomous task" realistic in 36 h on a 6 GB GPU?

### Takeaway
ACT is the only realistic choice. It is 52M params, uses about 211 MB at inference, works with about 50 demos, and is the most popular LeRobot policy. Recording 50 episodes takes about 30–45 min. Training for 100k steps takes 1.5 h on an A100 and about 7 h on a DGX Spark GB10. On a 2060 expect most of a night, or cut to 20–50k steps. It fits in 36 h only if data collection starts by about hour 12, and the hand-teleop demos must be smooth and consistent. Diffusion Policy is heavier to train and slow to sample, pi0 does not fit in 6 GB, and SmolVLA is 450M params, so fine-tuning is borderline on 6 GB.

### Cited Findings
- ACT is popular because of its "small size and fast inference speed" and its "straightforward application to limited amount of real-world demonstrations, allowing users to obtain well-performing policies with as little as 50 real-world trajectories". It is single-task and needs retraining when conditions change — [LeRobot paper](https://arxiv.org/pdf/2602.22818)
- LeRobot v0.3.3 ACT defaults: `chunk_size=100`, `n_action_steps=100`, `vision_backbone="resnet18"`, `dim_model=512`, `n_encoder_layers=4`, `n_decoder_layers=1` (a comment notes the original ACT's 7 decoder layers had a bug, so only 1 is effectively used), `use_vae=True`, `temporal_ensemble_coeff=None` (0.01 when ensembling), `optimizer_lr=1e-5` — [configuration_act.py @ v0.3.3](https://raw.githubusercontent.com/huggingface/lerobot/v0.3.3/src/lerobot/policies/act/configuration_act.py)
- **Classmethod SO-ARM101 ACT write-up (2026):** 50 episodes (40,075 frames, 840 MB) collected in about 45 min. Training ran 100k steps at batch size 8 in about 7 h (~3.9 steps/s) on a DGX Spark GB10, using about 18 GB of GPU memory. Success was 9/10 with CUDA inference at ~30 Hz versus 4/10 with Mac MPS inference at ~15 Hz, where the lower inference rate caused jerky motion. Pitfalls: `display_data=true` dropped recording from 30 Hz to 9.8 Hz, `uv add lerobot` pulled CPU-only torch, and `--policy.type` with `pretrained_path` skipped pre/post-processing (use `--policy.path`) — [Classmethod blog](https://dev.classmethod.jp/en/articles/lerobot-so-arm101-act-training-eval/)
- LeRobot Colab: about 1.5 h on an A100 at batch size 64 for 100k steps. LeRobot FAQ: 2–4 h for 100k ACT steps on PushT on one GPU — [LeRobot notebooks doc](https://gitea.d-robotics.cc/chao.wu/Lerobot/src/commit/0938a1d816c70d689ab6df3298a71572798c15a6/docs/source/notebooks.mdx); [LeRobot FAQ (mintlify mirror)](https://www.mintlify.com/huggingface/lerobot/resources/faq)
- Community SO-101 ACT checkpoints: one used 12 episodes (6,432 frames) for 20k steps. Another was fine-tuned for 20k steps at batch 8 in 2 h 22 m (2.54 steps/s) on an AMD Radeon GPU — [AaronL20/So101-act-test](https://huggingface.co/AaronL20/So101-act-test); [search result summary of HF SO-101 ACT model cards](https://huggingface.co/ddduk/so101_act35_real_sim_v1) (individual card details not re-verified)
- A first-timer guide plans 50 teleop episodes in 30–40 min — [Trelis: Train an ACT policy for SO-101](https://trelis.substack.com/p/train-an-act-policy-for-an-so-101)
- LeRobot Worldwide Hackathon: 14–15 June 2025, 100+ cities, more than 2,500 participants, centered on SO-101 — [CRIStAL Lille news](https://www.cristal.univ-lille.fr/en/news/2025-06-19-982/); [mind-verse recap](https://mind-verse.de/news/lerobot-worldwide-hackathon-2025-innovation-karrieren-ki-robotik)
- pi0 needs 13.32 GB of GPU memory in fp32 at inference, and SmolVLA needs 1.75 GB — [LeRobot paper, Table 2](https://arxiv.org/pdf/2602.22818)

### Inferences
- **Training throughput on a 2060:** the DGX Spark GB10 managed ~3.9 steps/s at batch 8, and an AMD consumer GPU 2.54 steps/s. A 2060 is plausibly ~2–4 steps/s with AMP fp16 (no bf16 on Turing). That is about 3–7 h for 50k steps and 7–14 h for 100k. The Classmethod run used ~18 GB, but on unified memory, which over-reports. On 6 GB, batch 8 with ResNet-18 at 480x640 ×1–2 cameras may need smaller images, one camera or batch 4. All of this is extrapolated and must be tested early.
- **Timeline that fits 36 h:** about h0–12 get hand → IK → arm teleop stable, h12–14 record 50 episodes (LeRobot dataset format, `display_data=false`), h14–24 train overnight (checkpoint every 10k steps), h24–30 evaluate and iterate, h30–36 demo. This is risky but realistic. The hand-teleop pipeline must log in LeRobot dataset format, with joint positions as the action and the 50 Hz command stream.
- **Hand-teleop demo quality is the main risk.** IK from a noisy hand produces jittery, inconsistent demos, and ACT learns that noise. Smoothing (One-Euro filter) before recording helps.
- **Diffusion acceleration angle (if the team wants kernels + policies):** train DP with a DDIM scheduler, or set `num_inference_steps=10` at inference. Then benchmark ladder: fp32 eager → AMP fp16 → `torch.compile(mode="reduce-overhead")` (CUDA graphs) → TensorRT export of the 1D-conv U-Net → custom fused kernel for the scheduler step plus GroupNorm+Mish. The denoising loop is about 10–100 sequential small U-Net calls at batch 1, so it is launch-overhead dominated, which CUDA graphs address directly. For ACT, CUDA graphs and fp16 would cut the ~5–30 ms more modestly.
- Compared with TensorRT/ORT/torch.compile, a hand-written kernel only wins on fusing the scheduler arithmetic and the elementwise tails. The convolutions should stay in cuDNN/TensorRT.

### Gaps
- No public report found of ACT trained on an RTX 2060 or other 6 GB card (VRAM use, steps/s).
- No official results or winners list from the June 2025 LeRobot hackathon, and no writeups specifically on ACT success rates there.
- Whether hand-tracking teleop (rather than leader-arm teleop) demos are good enough for ACT: no source found.

---

## Q5. Triton on Turing, CuPy / Numba alternatives, and CUDA-on-WSL2 limits

### Takeaway
Triton on sm_75 is "best effort". Since Triton 3.3, upstream lists compute capability 8.0+ as supported. Per PR #5066, `tl.dot` on compute capability < 8.0 falls back to FMA, i.e. no tensor cores, and bf16/fp8 are unsupported on Turing. Triton still works for elementwise or fused memory-bound kernels, but not for a tensor-core matmul story on a 2060. CuPy `RawKernel` (plain CUDA C) is the most predictable route on Turing. Numba's CUDA target has moved to the separate `numba-cuda` package. WSL2 works for CUDA, but pinned memory is limited, there is no full Unified Memory, and profiler support is partial (Nsight Systems CLI is listed as supported).

### Cited Findings
- Triton PR #5066, "[BACKEND] Deprecate MMAv1; fallback to FMA for computeCapability < 80", merged 5 Nov 2024. Maintainers said older platforms "won't break" but support is best-effort and untested. The FMA path "will still be maintained". The README's supported hardware was updated to CC 8.0+ — [triton-lang/triton PR #5066](https://github.com/triton-lang/triton/pull/5066)
- triton-windows README: Turing (GTX 16xx/RTX 20xx, sm75) is "officially supported by Triton <= 3.2" and "Support for Turing has been dropped since Triton 3.3". Official Triton does not support bf16 or fp8 on Turing, and triton-windows added fp8 since 3.2.0.post21 — [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)
- TileLang keeps native MMA lowering for FP16/INT8/INT4 on sm_75 (TITAN RTX). Its FP32/BF16 fallback is slower, and FP16 accumulation loses accuracy over long K — [tile-ai/tilelang PR #2811](https://github.com/tile-ai/tilelang/pull/2811)
- Numba: the built-in CUDA target is deprecated since 0.61 and maintained in the separate `numba-cuda` package (same `numba.cuda` namespace, `pip install numba-cuda`). Numba 0.63+ warns without it, and the built-in target is to be removed no earlier than 0.63 — [Numba deprecation notices](https://numba.readthedocs.io/en/latest/reference/deprecation.html); [Numba CUDA overview](https://numba.readthedocs.io/en/stable/cuda/overview.html)
- CUDA on WSL2 (NVIDIA user guide):
  - Pinned system memory has limited availability, and some workloads exceeding it may fail.
  - "Full Managed Memory Support is not available", "UVM full features will not be available". Managed memory may run slower with high system memory use, and concurrent CPU/GPU access is not supported.
  - NVML/nvidia-smi does not support all queries (e.g. GPU utilization, per-process queries).
  - The support table lists Nsight Systems CLI and CUPTI trace as supported on Volta+, with profilers tied to specific Windows/driver builds. The page states that profilers are "not available yet" in a preview note and does not mention Nsight Compute.
  - Overhead is described as near-native.
  - [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Inferences
- **Practical implication for torch.compile on a 2060:** Inductor generates Triton kernels. Elementwise and reduction fusions should still work on sm_75 (best effort), but Triton-generated matmuls would not use tensor cores, so mm/conv should stay on cuBLAS/cuDNN (Inductor's default for most shapes). Pin a known-working PyTorch + Triton pair early and smoke-test `torch.compile` on the 2060 in the first hour. Not verified on hardware.
- For a fp16 tensor-core story on Turing, use cuBLAS/cuDNN/TensorRT, or CUDA C WMMA (`mma.sync` m16n8k8 on sm_75) via CuPy RawKernel. Triton ≥3.3 will not give it.
- **Profiling in WSL2:** plan on `nsys` CLI plus CUDA events / `torch.cuda.Event` timing for before/after numbers. Do not rely on Nsight Compute kernel-level metrics working in WSL2. Avoid `cudaMallocManaged`-based designs, and keep pinned host buffers small (one frame).
- **Recommended tool for a student team:** CuPy RawKernel (CUDA C string, NVRTC-compiled, zero-copy interop with PyTorch via DLPack) for preprocessing or scheduler-step kernels. Triton is fine for elementwise fusions if the installed version works on sm_75.

### Gaps
- I did not confirm, beyond the PR title and maintainer comments, whether PR #5066 also removed Turing's MMAv2 (mma.sync) path for `tl.dot`. The title says FMA fallback for CC < 80, which implies yes, but the diff was not inspected.
- Which PyTorch release first bundled a Triton ≥3.3 (believed 2.7), and any concrete torch.compile failures on T4 or RTX 20-series: no confirming source found.
- Nsight Compute support status on WSL2 in 2026 was not established from the NVIDIA page.
