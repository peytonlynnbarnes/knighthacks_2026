# Building, Benchmarking and Presenting Custom GPU Kernel Work for a Hand→SO-ARM101 Teleop Hackathon (RTX 2060 / sm_75 / WSL2)

Scope note: about 15 tool calls. Items marked "(background knowledge, unverified this session)" are not backed by a fetched source and should be checked or rephrased by the report writer.

## 1. Kernel-optimization methodology and how small-matrix robotics kernels map to GPUs

### Takeaway
The progression judges recognize is the one Simon Boehm's SGEMM worklog made standard: naive → coalesced → shared-memory caching → register/block tiling → vectorized loads → autotuning → warp tiling, with the % of a reference library reported at each step. Robotics kernels made of tiny matrices (4×4 transforms, 6×5 Jacobians, 5×5 solves) are a different problem from big GEMMs. They are latency- and launch-bound, not FLOP-bound. Every robotics reference here (cuRobo, GRiD, MPCGPU, MAGMA batched) gets its speedup by batching many independent problems, using a few threads per small matrix, and cutting launch overhead with fusion and CUDA graphs.

### Cited Findings
- Boehm's worklog (Dec 2022) starts from a naive FP32 SGEMM kernel and applies one optimization at a time, aiming for about 95% of cuBLAS. The steps are: K1 naive (~1.3% of cuBLAS), K2 GMEM coalescing, K3 SMEM caching, K4 1D blocktiling, K5 2D blocktiling, K6 vectorized memory access, K9 autotuning, K10 warptiling (~93.7%). It uses CUDA cores only, no tensor cores. — [Boehm worklog (gwern mirror)](https://gwern.net/doc/www/siboehm.com/d4b3e9d6de0655f490fe6369f4d004936248a833.html)
- Modular's matmul custom-op tutorial follows steps 1–6 of Boehm's progression, so it is a second worked example of the same ladder. — [Modular docs](https://docs.modular.com/stable/develop/custom-ops-matmul/)
- **cuRobo kernel design** (tech report, arXiv 2310.17274):
  - It stores transforms as 4×4 homogeneous matrices so that "four parallel threads" compute each matrix multiply. The FK backward pass uses "16 threads to read and project the gradients from Cartesian space to joint space." It gives several threads to each query to hide memory overhead. — [cuRobo tech report](https://arxiv.org/html/2310.17274)
  - It records optimizer iterations as CUDA Graphs and replays them with new data. "This use of CUDA Graphs reduced our planning time by 10x compared to calling the kernels individually from python." Remaining Python overhead was about 8 ms mean and 30 ms p98. — [cuRobo tech report](https://arxiv.org/html/2310.17274)
  - IK runs 30 seeds in parallel per query and keeps the best one. At batch 1000, plain IK takes 27 ms (about 37k solutions/s) against TracIK's 629 ms, a 23.4× speedup. Collision-free IK at batch 1000 takes 131 ms (about 7.6k/s). — [cuRobo tech report](https://arxiv.org/html/2310.17274)
  - **At batch 1, cuRobo IK (2.7 ms) is slower than CPU TracIK (0.9 ms).** cuRobo only pulls ahead from batch size about 10. — [cuRobo tech report](https://arxiv.org/html/2310.17274). An NVIDIA patent reports a similar single-query result (3.64 ms vs 2 ms). — [USPTO patent 12496714](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12496714)
  - Collision checking uses spheres. Self-collision checks sphere pairs. World collision uses analytic cuboids, NVIDIA Warp's BVH for meshes, and nvblox ESDF for depth. Continuous checks run one thread per trajectory timestep. — [cuRobo tech report](https://arxiv.org/html/2310.17274)
  - NVIDIA's blog says cuRobo ships "a library of custom robotics CUDA kernels" on top of PyTorch and uses CUDA Graphs to cut launch overhead. It was benchmarked on an RTX 4090 and on AGX Orin, where motion plans take under 100 ms median. — [NVIDIA blog](https://developer.nvidia.com/blog/cuda-accelerated-robot-motion-generation-in-milliseconds-with-curobo)
- **GRiD** (Plancher et al.) generates dynamics kernels from the URDF. It exploits fine-grained parallelism inside each computation and coarse-grained parallelism across computations. It is up to 7.2× faster (v1; 7.6× in a later version) than a multi-threaded CPU, but **only up to 2.5× (2.6×) once I/O overhead is included**. That drop is a ready-made slide about transfer cost. — [arXiv 2109.06976](https://arxiv.org/abs/2109.06976); [export.arxiv abstract](https://export.arxiv.org/abs/2109.06976)
- **MPCGPU** (Adabag, Atal, Gerard, Plancher, ICRA 2024) is GPU nonlinear MPC built on a custom preconditioned-conjugate-gradient solver. It reaches kHz control rates on a Kuka IIWA tracking task with up to 512 knot points and beats a CPU solver by ≥10× on most solves (3.6× on average). The authors note that iterative solvers have variable solve time. — [arXiv 2309.08079](https://arxiv.org/pdf/2309.08079)
- **Thread-per-matrix vs warp-per-matrix:**
  - Thread-per-matrix is generally used only for the smallest sizes (n ≤ 8), because each thread holds n² values in registers and larger sizes spill to local memory. — [arXiv 2609.15217](https://arxiv.org/pdf/2609.15217)
  - Warp-per-matrix LU (Wainwright 2013) works up to 32×32. — [same source](https://arxiv.org/pdf/2609.15217)
  - MAGMA batched LU/QR/Cholesky for n ≤ 32 (Abdelfattah et al.) argues that "a different mindset" is needed for sub-warp-sized matrices and reports up to 11.8× over vendor libraries on V100. — [ICL UTK](https://icl.utk.edu/node/1209)
  - Counterpoint: one paper found a single-thread batched LU no faster than cuBLAS batched LU. — [arXiv 2609.15217](https://arxiv.org/pdf/2609.15217)
- **Shared memory isn't automatically a win.** A 2026 multibody FEM GPU paper found that staging data in shared memory lowered occupancy and did not help. — [arXiv 2604.10357](https://arxiv.org/pdf/2604.10357)

### Inferences
- For a 5-DOF arm, a whole problem is tiny: 6 transforms, a 6×5 Jacobian, and a 5×5 (JᵀJ+λI) solve. **Thread-per-seed is the natural first step:**
  - About 25 floats of state per seed fit easily in Turing's 255 registers/thread.
  - With an SoA layout (`q[joint][seed]`), consecutive threads read consecutive addresses, so loads coalesce.
  - The optimization ladder can then be:
    - v0: AoS layout, one kernel per DLS iteration, launched from Python.
    - v1: SoA layout.
    - v2: fuse FK, Jacobian, solve and update into one kernel.
    - v3: loop all N iterations inside the kernel and keep the joint state in registers.
    - v4: warp-shuffle argmin to pick the best seed on the GPU, then copy back only 5 floats.
    - v5: CUDA graph or persistent kernel.
    - v6: try 4 or 16 threads per seed, cuRobo style, and compare.
- Each step should be shown with Nsight Compute metrics (achieved occupancy, memory throughput %, registers/thread). That makes the "technical depth" visible.
- Roofline framing for presenting: compute arithmetic intensity (FLOPs/byte) for the fused kernel. A full FK+Jacobian+solve per seed is a few thousand FLOPs over about 20–40 bytes of I/O, so it should sit far to the right (compute side). The unfused AoS version sits on the memory-bound slope. Nsight Compute has a built-in roofline chart in its "GPU Speed Of Light Roofline" section (background knowledge, unverified this session).
- RTX 2060 peak numbers for the roofline axes: about 6.5 FP32 TFLOPS, 336 GB/s GDDR6, 30 SMs (background knowledge, unverified this session).

### Gaps
- I didn't fetch a dedicated roofline tutorial. Good candidates are Williams/Waterman/Patterson 2009 and NVIDIA's Nsight Compute roofline docs.
- I didn't get GRiD's exact block/thread mapping. The abstract only says "fine-grained within, coarse-grained across" computations.
- cuRobo appendix E (per-kernel timings) was not read; it starts at character offset 100000 of the HTML version.

## 2. Tools for writing CUDA from Python on sm_75

### Takeaway
On an RTX 2060 in 2026, the safe choices are **CuPy RawKernel** (raw CUDA C, the fastest route to a "real kernel" story), **NVIDIA Warp** (Python kernels, robotics-oriented, still supports sm_75), and **Numba CUDA through the `numba-cuda` package**. Avoid **Triton ≥ 3.3**, which dropped Turing; pin 3.2.x if you need it. PyTorch `load_inline` works but costs minutes of compile time on every rebuild.

### Cited Findings
- **NVIDIA Warp:**
  - Builds against CUDA 12.x support sm_52 (Maxwell) and newer. Builds against CUDA 13.x require **sm_75 (Turing) or newer**, so the RTX 2060 is exactly at the floor.
  - Driver minimums: ≥525 for CUDA 12.x builds, ≥580 for CUDA 13.x builds.
  - The minimum is set by the CUDA toolkit the Warp wheel was built with, not by the toolkit installed on your system.
  - Requires Python 3.10+.
  - fp16 atomics need CC ≥ 7.0.
  - [Warp compatibility docs](https://nvidia.github.io/warp/latest/user_guide/compatibility.html)
- cuRobo itself uses Warp's BVH for mesh collision queries, which shows Warp is used in real robotics kernels. — [cuRobo tech report](https://arxiv.org/html/2310.17274)
- **Triton:**
  - Turing (sm_75) is supported officially only up to Triton 3.2. Support was dropped from Triton 3.3 (triton-lang issue #5066, as cited by the triton-windows fork).
  - Official Triton has no bf16 or fp8 support on Turing.
  - This is a secondary source; the official release notes were not checked.
  - [triton-windows README](https://github.com/woct0rdho/triton-windows)
- **Numba:**
  - The CUDA target built into Numba is deprecated, and development moved to NVIDIA's `numba-cuda` package. The `numba.cuda` namespace and existing code stay the same.
  - Install with `pip install numba-cuda`.
  - [Numba CUDA overview](https://numba.readthedocs.io/en/stable/cuda/overview.html); [Deprecation notices](https://numba.readthedocs.io/en/latest/reference/deprecation.html)
- cuRobo is "implemented in PyTorch" with custom CUDA kernels as extensions, the PyTorch custom-op pattern. — [NVIDIA blog](https://developer.nvidia.com/blog/cuda-accelerated-robot-motion-generation-in-milliseconds-with-curobo)

### Inferences (pros/cons for 36 h; partly background knowledge)
| Tool | Pros | Cons |
|---|---|---|
| CuPy `RawKernel` / `RawModule` | Real CUDA C (shared memory, `__shfl_sync`, `__launch_bounds__`); NVRTC compiles in about 1 s and caches; NumPy-like arrays; Nsight shows the kernel by name; supports streams and CUDA graphs | Write C strings; debugging is printf |
| Numba CUDA (`numba-cuda`) | Pure Python syntax, `cuda.shared.array`, `cuda.local.array`, warp intrinsics | JIT first-call latency; less control over registers; package migration churn |
| NVIDIA Warp | Python kernels with `wp.mat44`/`wp.transform` types, autodiff, `wp.ScopedCapture` for CUDA graphs, robotics/sim examples | Its own abstraction, so "hand-optimized kernel" is less obvious to judges; the sm_75 floor on CUDA 13 builds |
| Triton | Block-level programming, good for fused elementwise or reduction work | Turing dropped after 3.2; designed for big tiled tensors, a poor fit for tiny per-thread problems |
| PyTorch `cpp_extension.load_inline` | Integrates with torch tensors and autograd | Slow compile (tens of seconds to minutes), heavy dependency |
| JAX | `vmap` over seeds is elegant | Can't hand-write kernels (Pallas targets newer GPUs/TPU); XLA hides the optimization story |

- Recommendation: use **CuPy RawKernel** for the hand-written kernel ladder (best "depth" story), with a NumPy CPU reference for correctness and timing.
- Warp is a strong option if the team prefers Python syntax. Check the installed wheel's CUDA build version against the Windows driver (≥580 for CUDA 13 wheels).

### Gaps
- I didn't verify CuPy's or JAX's current minimum compute capability in docs. It is believed fine for sm_75 (background knowledge).
- I didn't verify whether Warp examples include IK or MPPI specifically.

## 3. Profiling and benchmarking under WSL2

### Takeaway
Both Nsight tools work in WSL2 today.
- **Nsight Compute (kernel counters):** needs Windows 11 with driver ≥525, or Windows 10 build ≥19044 with driver ≥545. **"Access to NVIDIA GPU Performance Counters" must also be enabled in the Windows-side NVIDIA Control Panel.**
- **Nsight Systems (timeline):** supported as a preview.

Use CUDA events plus warmup for kernel timing, wall-clock for host-inclusive timing, and report p50/p99 rather than the mean.

### Cited Findings
- Nsight Compute system requirements (v2026.2.1 docs):
  - "Profiling on WSL is only supported with WSL version 2."
  - Windows 10: OS build ≥19044 and driver ≥545 (and not on systems with >1 TB RAM).
  - Windows 11: driver ≥525.
  - "Access to NVIDIA GPU Performance Counters must be enabled in the NVIDIA Control Panel of the Windows host."
  - Nsight Compute "requires a CUDA driver that is compatible with CUDA 13."
  - [Nsight Compute system requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)
- WSL2 profiling first arrived in Nsight Compute 2022.4, and Windows 10 support in 2023.3. Install the **Linux x86_64** Nsight Compute package inside WSL2. — [Nsight Compute 2022.4 announcement](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2022-4-is-now-available/236195); [Updates 2023.3](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2023-3.html)
- Common failure: `==ERROR== Profiling is not supported on device 0 as it uses the Windows Subsystem for Linux (WSL)`. An NVIDIA moderator's fix was to upgrade to Nsight Compute ≥2023.3. — [NVIDIA forum](https://forums.developer.nvidia.com/t/error-profiling-is-not-supported-on-device-0-as-it-uses-the-windows-subsystem-for-linux-wsl/260814)
- CUDA on WSL user guide:
  - Supported as preview: "Nsight Systems CLI, and CUPTI (Trace) – Volta and later"; "Developer tools – Profilers – Volta and later" (Win10 19044+ with r545+, or Win11 with r525+); Compute Sanitizer, Pascal and later.
  - Limits: pinned memory "availability for applications is limited", and no concurrent CPU/GPU access to managed memory.
  - Some `nvidia-smi` NVML queries ("GPU utilization, active compute process") are not supported.
  - The guide contradicts itself: section 4 still says "profilers are not available yet", while the table lists them as supported.
  - [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Inferences
- **Setup checklist:**
  1. Windows NVIDIA Control Panel → Developer → Manage GPU Performance Counters → "Allow access to all users".
  2. Install a recent Linux Nsight Compute (`ncu`) inside WSL2.
  3. Run `ncu --set full -k <kernel> python bench.py` and save a `.ncu-rep`. Open it in the Windows GUI for roofline and occupancy screenshots.
  4. Use `nsys profile -t cuda,nvtx,osrt` for the end-to-end timeline: camera → MediaPipe → IK kernel → ROS publish.
  5. Don't rely on `nvidia-smi` utilization inside WSL.
- **Benchmark protocol (standard practice; background knowledge):**
  - Warm up (JIT/NVRTC compile, first-launch context init) for about 50–100 iterations.
  - Time the kernel alone with `cudaEvent` (CuPy `cp.cuda.Event` / `cp.cuda.get_elapsed_time`).
  - Time host-inclusive with `time.perf_counter_ns()` around launch + sync + D2H copy.
  - Collect ≥1000 samples and report p50/p99/max. Fix the clocks or note GPU boost behaviour.
  - Plot batch size vs latency on a log-log axis for CPU (NumPy and C), GPU-naive and GPU-optimized. The crossover point is the key slide (see §5).
- **Glass-to-motion latency:**
  - Film the hand and the arm together with a phone in 240 fps slow-mo (about 4.2 ms per frame). Count frames from hand-motion onset to arm-motion onset.
  - Repeat about 10 times and report the median.
  - Keep a split "pipeline budget" bar (camera exposure + MediaPipe + IK + ROS + servo bus + servo response) from per-stage timestamps.
  - The GPU kernel will likely be <1% of glass-to-motion. Present that honestly: the GPU's value is many seeds or MPPI samples within the same budget, not lower latency.
- **Before/after visuals:**
  - Optimization ladder bar chart (µs per step, log scale) with % of roofline annotated.
  - Nsight Compute side-by-side "Speed of Light" screenshots.
  - Batch-size crossover plot.
  - A live on-screen overlay of IK latency p50/p99 during the demo.

### Gaps
- I didn't confirm whether the current Nsight Systems release has moved WSL2 support out of "preview". The WSL guide still lists it as preview.
- I found no source on GPU clock locking (`nvidia-smi -lgc`) inside WSL2. It is likely unsupported since NVML is limited there; test it.

## 4. Prior art and what hackathon judges reward

### Takeaway
The published vision-based teleop systems (AnyTeleop, Bunny-VisionPro) are bottlenecked by **hand-pose detection on the GPU**, not by retargeting or IK. Retargeting and IK run in about 1–3 ms on CPU. A GPU-kernel story is therefore credible only if framed as "more robustness per millisecond": multi-seed IK that avoids local minima, MPPI or collision checking. It is not credible framed as "made the 1 ms IK faster". Hackathon guides consistently rank a working live demo above slides, and want the hardest technical decision named explicitly.

### Cited Findings
- **AnyTeleop** (2023):
  - Hand pose detection on the GPU is the most time-consuming module, and its designed maximum is 25 Hz. Retargeting runs at the same rate.
  - The authors found it hard to sustain that throughput with every module on one computer and suggest running control on a separate machine.
  - [arXiv 2307.04577](https://arxiv.org/pdf/2307.04577)
- **Bunny-VisionPro** (2024):
  - CPU timings on an i7-12700KF: retargeting 2.54 ms (3.43 ms with its loop-joint method), IK-only motion control 0.74 ms.
  - The full pipeline runs in real time at >60 Hz with singularity and collision avoidance.
  - [arXiv 2407.03162](https://arxiv.org/html/2407.03162v1)
- A third-party aggregator lists AnyTeleop retargeting at 15.59 ms mean latency on a Wuji hand. **Unverified**; may come from a later comparison paper. — [wizwand](https://www.wizwand.com/paper/68fa73311e209131cea5b05c)
- "Geometric Retargeting" (2025) presents an "ultrafast neural hand retargeting" approach, which is more prior art on retargeting speed. — [arXiv 2503.07541](https://arxiv.org/pdf/2503.07541)
- Open-TeleVision: I found no latency or GPU figures.
- **LeRobot Worldwide Hackathon** (June 2025):
  - Ran in 100+ cities on SO-100/101 arms.
  - Each team submitted its recorded dataset and a ≤2 minute video.
  - [Mind-Verse report](https://mind-verse.de/news/lerobot-worldwide-hackathon-2025-innovation-karrieren-ki-robotik); [Betaworks NYC event](https://www.betaworks.com/event/nyc-lerobot-hackathon); [Hackathon README](https://lerobot-worldwide-hackathon-readme.static.hf.space/README.md)
- AMD Open Robotics Hackathon (Tokyo/Paris, SO-101 + LeRobot): all projects were built during the event and documented in public GitHub repos. — [AMD recap](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-open-robotics-hackathon-recap.html)
- **Judging:**
  - Devfolio criteria: technicality, originality, practicality, usability and "wow factor". It advises showing the project working rather than slides, and not speeding up demo videos to the point of being unclear. — [Devfolio judging guide](https://guide.devfolio.co/docs/guide/participating-in-hackathons/judging-1)
  - Judges value meaningful innovation over complex stacks. A rough working build scores above polished slides. Name the hardest engineering decision out loud. Spend about half the slot on the demo. — [UoKufa judging criteria guide](https://hackathon.uokufa.edu.iq/en/guides/judging-criteria-explained/); [AngelHack demo tips](https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/)
  - A HackerEarth post claims to draw on the MLH Organizer Guide. I didn't retrieve the MLH guide itself. — [HackerEarth](https://www.hackerearth.com/blog/10-tips-win-hackathon)

### Inferences
- **Pitch framing:** "MediaPipe hand → 5-DOF IK at 50 Hz. On the GPU we solve 4,096 IK seeds (or 1,024 MPPI rollouts with collision costs) in X µs, against Y ms on CPU. That removes IK flips and singularity jumps, as shown in a side-by-side video."
  - The metric judges can see is jitter or failure rate, not just µs.
  - Show the optimization ladder as backup depth.
- **Demo reliability:** keep a CPU fallback path (single-seed DLS IK) behind a flag, so a CUDA or WSL hiccup can't kill the live demo.

### Gaps
- I found no hackathon writeup specifically about adding GPU kernel optimization to a robotics teleop project.
- I found no list of LeRobot hackathon winners.
- I found no GPU-accelerated MPPI teleop hackathon demo in this session. Candidates to check: pytorch_mppi, NVIDIA Isaac Lab / cuRobo MPC examples, the "STORM" MPC by Bhardwaj et al. 2021.

## 5. Pitfall: launch and transfer overhead for tiny 50 Hz workloads

### Takeaway
At 50 Hz (a 20 ms budget), one tiny IK problem takes about 1 µs of math, but launch, sync and copies cost about 5–20 µs per kernel, plus Python overhead. **The GPU loses at batch 1.** cuRobo IK takes 2.7 ms against TracIK's 0.9 ms on CPU, and only wins from batch ≥10. GRiD's 7.2× speedup shrinks to 2.5× once I/O is counted. The fixes are to batch (many seeds or rollouts per call), fuse into one kernel, capture a CUDA graph, and copy back only the winner.

### Cited Findings
- NVIDIA's CUDA Graphs blog (V100) measured three cases for a short kernel:
  - Synchronizing after each launch: about 9.6 µs per kernel against 2.9 µs of execution.
  - Graph replay: about 3.4 µs effective per kernel.
  - One-time graph creation: about 400 µs, amortized over replays.
  - [NVIDIA blog: CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs)
- An NVIDIA forum post: null-kernel launch around 5 µs on PCIe gen3 "has been the standard for the past decade". — [NVIDIA forum](https://forums.developer.nvidia.com/t/launch-of-many-small-kernels-10x-slower-compared-to-one-kernel/350194)
- NVIDIA also has a guide on using CUDA graphs when parameters change each iteration (graph update, or exec-update). — [NVIDIA blog: Employing CUDA Graphs in a Dynamic Environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/)
- cuRobo: CUDA graphs gave a 10× planning-time reduction over individual kernel calls from Python. Batch-1 IK is 3× slower than CPU TracIK; the GPU only wins at batch ≥10. — [cuRobo tech report](https://arxiv.org/html/2310.17274)
- GRiD: 7.2× without I/O falls to 2.5× with I/O. — [arXiv 2109.06976](https://arxiv.org/abs/2109.06976)
- WSL2-specific: pinned-memory availability is limited, and managed memory performance is reduced with no concurrent CPU/GPU access. Prefer explicit small pinned buffers and avoid Unified Memory. — [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Inferences
- **Rough per-tick budget at 50 Hz (estimate):**
  - Python → CuPy launch: about 10–30 µs per kernel.
  - H2D copy of the 3-float target: about 10 µs.
  - D2H copy of 5 joint angles, with sync: about 10–20 µs.
  - Total GPU round trip: ≥50–100 µs, against a C/NumPy single-seed DLS IK of about 10–100 µs.
  - So the GPU only makes sense when it does **work the CPU can't**: N ≥ ~10³ seeds, MPPI with K×T rollouts, or dense collision checks.
  - WSL2 adds a virtualization (GPU-PV) layer, so launch latency may be higher than native Linux. **Measure it**: a null-kernel launch microbenchmark is a good first slide.
- **Mitigations, in order of payoff:**
  1. Fuse everything into one kernel per control tick.
  2. Iterate inside the kernel rather than relaunching per iteration.
  3. Reduce on the GPU (warp shuffle, then block argmin) and copy back only the best solution.
  4. Capture as a CUDA graph (CuPy `stream.begin_capture()`/`end_capture()`, Warp `wp.ScopedCapture`, or `torch.cuda.CUDAGraph`). Write new targets into a fixed device buffer before replay.
  5. Use pinned host buffers and async copies on a dedicated stream.
  6. Advanced: a persistent kernel that polls a mapped host flag, which removes launch cost entirely. This is risky under WSL2 because pinned/mapped memory is limited and watchdog/TDR rules apply on Windows display GPUs. Treat it as a stretch goal.
- **Honest crossover plot (latency vs batch size, CPU vs GPU)** is both the best defence against the "why GPU?" question and an easy technical-depth win.

### Gaps
- I found no published measurement of CUDA kernel launch latency specifically under WSL2 vs native Linux on Turing. The team should measure it.
- Windows TDR (2 s default GPU timeout) applies to WSL2 kernels on a display GPU. This is background knowledge, unverified this session, and matters only for long persistent kernels.
