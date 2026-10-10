# GPU-accelerated planning and control workloads for a 5-DOF hand-teleoperated SO-ARM101 (RTX 2060 6 GB, sm_75, WSL2)

Scope: batched/many-seed IK, sampling-based MPC (MPPI/STORM), collision checking, motion retargeting, other GPU-shaped planning work. The goal is to find what is useful (visible improvement) and doable in 36 h, and what a custom CUDA kernel would look like.

Local context, from `/home/peyton/kh/INTEGRATION.md`: the current pipeline is `/ball_position` (6 floats, offset from HOME, 30 Hz) → `ik_node.py` (EMA, workspace box, task-priority numpy IK, hold on unreachable/stale) → `/arm/<joint>` (5 joints, rad) → `feetech_mirror.py`. The arm has 5 DOF, so a full 6-DoF pose is generally unreachable. Any IK has to trade off position against orientation (task priority), and that trade-off is where solution "flips" come from.

Hardware caveat for every number below: almost all published timings are from an RTX 4090, Titan RTX, RTX 4060 Laptop or Jetson Orin, and from 7-DOF (Franka) or larger robots. None were measured on an RTX 2060 or a 5-DOF chain. The scaling inferences are mine and are marked as such.

---

## Q1. Batched / many-seed IK on GPU: seeds/sec, latency, prior art, custom kernel

### Takeaway
Batched GPU IK is mature. cuRobo reports about 37k IK solutions/s (30 seeds each) at batch 1000 on an RTX 4090, and HJCD-IK reports 8.6 ms for 2000 Panda IK problems on an RTX 4060 Laptop under WSL. At batch size 1, however, GPU IK is **slower** than CPU (cuRobo 2.7 ms vs TRAC-IK 0.9 ms). A single 5-DOF IK target at 30–50 Hz is not compute-bound. The useful version is "many seeds per frame, then pick the best-scoring, continuity-preserving solution", which directly attacks solution flips. A hand-written FK+Jacobian+DLS kernel for a 5-joint chain is small (about 150 lines of CUDA), fits in registers, and is a good showcase kernel.

### Cited Findings
- cuRobo IK vs TRAC-IK: at batch 1, cuRobo takes 2.7 ms and TRAC-IK 0.9 ms. cuRobo becomes faster from batch 10 upward. At batch 1000, cuRobo takes about 27 ms (about 37,134 solutions/s) and TRAC-IK about 629 ms (about 1,590/s). Each query uses 30 parallel seeds (Halton-sampled) and returns the best. — [cuRobo paper, arXiv 2310.17274](https://arxiv.org/html/2310.17274v2)
- cuRobo collision-free IK: batch 1000 in about 131 ms (about 7,611 solutions/s) vs about 95/s for rejection-sampled TRAC-IK+PyBullet. Rejection sampling failed on 20% of problems. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- cuRobo's motion-generation pipeline uses 500 IK seeds. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- cuRobo hardware: Ryzen 9 7950X + RTX 4090, and Jetson AGX Orin 64GB. A reachability demo ran 500 poses at about 15 Hz while sharing the GPU with Isaac Sim. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- cuRobo kernel notes: kinematics runs as a **single CUDA kernel**, versus about 125 kernels in STORM's PyTorch version. Transforms are stored as 4×4 matrices, four threads cooperate per matrix multiply, and the backward pass uses 16 threads to project Cartesian gradients to joint space. Kinematics costs about 1 ns/query at batch 100k and is up to 891× faster than CPU at 100k. CUDA Graphs cut geometric planning time by about 10×. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- HJCD-IK (2025): stage 1 is a batched coordinate-descent seed generator and stage 2 is a Levenberg-Marquardt polish. It was benchmarked on an **RTX 4060 Laptop under Windows 11 + WSL (Ubuntu 24.04), CUDA 12.5**, which is close to the team's setup. Mean times at batch 2000: HJCD-IK 8.58 ms (Panda) / 8.72 ms (Fetch), PyRoki 13.97 / 13.37 ms, cuRobo 30.30 / 14.62 ms, IKFlow 99.98 / 87.89 ms. At batch 1, HJCD-IK takes 5.18 ms (Panda). Kernel layout: **one thread block per seed** (not one thread per seed), two warps per joint in stage 1, and residuals in shared memory for greedy selection. — [HJCD-IK, arXiv 2510.07514](https://arxiv.org/html/2510.07514v1)
- IKFlow (normalizing flow): generates 2000 solutions in under 10 ms, accurate to within 10 mm and 2°, and needs a refinement step for exact solutions. It was trained on an RTX 2080 Ti, but the inference hardware for that timing is not stated in the snippets. — [IKFlow, arXiv 2111.08933](https://arxiv.org/pdf/2111.08933); [repo](https://github.com/jstmn/ikflow)
- PyRoki: MIT-licensed, JAX (CPU/GPU/TPU). It provides differentiable FK from URDF, a Levenberg-Marquardt solver (jaxls) with manifold support and augmented-Lagrangian constraints, and costs for pose, self/world collision (spheres, capsules, halfspaces, heightmaps) and manipulability. The README publishes no benchmarks, says PyRoki is likely slower than other toolkits in collision-heavy scenes, and notes JIT recompiles whenever input shapes change. — [PyRoki GitHub](https://github.com/chungmin99/pyroki)
- pytorch_kinematics offers batched FK, Jacobians and damped-least-squares IK, batched over goals and over retry seeds. I found no published GPU timings. — [pytorch-kinematics PyPI](https://pypi.org/project/pytorch-kinematics)
- At batch size 1, CPU beat GPU for amortized neural IK. The GPU only wins at larger batches, because small batches are overhead-bound. — [arXiv 2604.16629 (search snippet)](https://arxiv.org/pdf/2604.16629)

### Inferences
- **Scale for this arm:** the SO-101 has 5 revolute joints and the team solves one target per frame at 30–50 Hz. A numpy DLS solve is 5×5/6×5 linear algebra and probably takes well under a millisecond per iteration on CPU, so GPU throughput adds nothing for a single seed. The GPU pays off only if you run **hundreds to thousands of seeds per frame** and use them for quality:
  1. Global coverage: always find a solution when one exists, which removes the "random restarts on failure" stalls in the current code.
  2. **Flip avoidance:** score each converged seed by `w_err·‖task error‖ + w_cont·‖q − q_prev‖²_W + w_lim·limit_margin + w_manip·(−manipulability)` and pick the argmin. With 1k seeds covering the elbow-up/elbow-down and wrist branches, continuity becomes an explicit choice instead of whatever the warm start converges to.
  3. **Look-ahead IK**: solve for several predicted future hand positions in the same launch (batch = seeds × time offsets).
- **Expected RTX 2060 numbers (extrapolated, not measured):** a 5-DOF chain has about 5/7 the FK work of a Panda. One thread per seed, 20 DLS iterations, 4096 seeds: each iteration is about 5 small 4×4 transform compositions + a 6×5 Jacobian (analytic, from joint axes and positions) + a 5×5 solve (or 6×6 for J Jᵀ+λ²I), so a few thousand FLOPs per iteration and about 10⁸–10⁹ FLOPs per frame. The RTX 2060 has about 6.5 TFLOPS fp32 (spec-sheet figure, not cited here), so the kernel itself should run in **sub-millisecond to low-millisecond** time. Launch, copy and Python overhead (about 10–50 µs per launch, plus the CuPy/PyTorch call) will dominate. This is an inference; benchmark it.
- **Memory:** 4096 seeds × 5 joints × 4 B = 80 KB. Memory is a non-issue on 6 GB.
- **Minimal custom kernel (one thread per seed, fp32, everything in registers):**
  ```cuda
  // DH/URDF constants in __constant__ memory: per joint origin xyz/rpy -> precomputed 4x4 (3x4) fixed transform + axis
  __constant__ float T_fixed[5][12]; __constant__ float axis[5][3]; __constant__ float qlo[5], qhi[5];
  extern "C" __global__ void dls_ik(const float* q0, float* q_out, float* err_out,
                                    float3 p_tgt, float3 z_tgt, float lam, float w_ori, int iters, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    float q[5]; for (int j=0;j<5;++j) q[j] = q0[i*5+j];          // registers
    for (int it=0; it<iters; ++it) {
      float R[9]={1,0,0,0,1,0,0,0,1}, p[3]={0,0,0}, zj[5][3], pj[5][3];
      #pragma unroll
      for (int j=0;j<5;++j) { compose(R,p,T_fixed[j]); rot_axis(R, axis[j], q[j]); /* sincosf */
                              store joint axis zj[j]=R*axis, position pj[j]=p; }
      compose(R,p,T_tool);
      float e[6]; pos_err(e, p_tgt, p); ori_err(e+3, R, z_tgt, w_ori);  // 5-DOF: position + pointing (2-3 rows)
      float J[6][5]; for j: J[0..2][j]=cross(zj[j], p-pj[j]); J[3..5][j]=w_ori*zj[j];
      // dq = J^T (J J^T + lam^2 I)^-1 e   (6x6)  or  (J^T J + lam^2 I)^-1 J^T e (5x5, cheaper): Cholesky, unrolled
      float A[15], b[5]; jtj_plus_lambda(A,b,J,e,lam); chol5_solve(A,b);
      for j: q[j] = clamp(q[j]+b[j], qlo[j], qhi[j]);
    }
    // final cost incl. continuity term computed by caller or here; write q_out, err_out
  }
  ```
  Launch from CuPy `RawKernel` (or `cupy.RawModule` with `-use_fast_math`). Pick the best seed with `cupy.argmin` or a custom warp-reduce kernel.
- **Optimization steps that make a good demo story** (each measurable with Nsight Compute, `--ptxas-options=-v` and `cupy.cuda.Event` timings):
  1. Baseline: numpy CPU loop over N seeds, then a PyTorch batched version (many small kernels, like STORM's "125 kernels").
  2. Fused single kernel (FK + Jacobian + solve + update + all iterations in one launch). This is cuRobo's main win ("single CUDA kernel").
  3. Constants (link transforms, limits) in `__constant__` memory, `#pragma unroll` over the 5 joints, and `sincosf`/`__sinf` (fast-math) to keep everything in registers. Check for 0 bytes of local-memory spill with `-Xptxas -v`. sm_75 allows 255 registers/thread; target ≤64 for full occupancy at 1024 threads/SM. Turing has 64K registers/SM, which is spec knowledge, not cited.
  4. Exploit the 5-DOF structure: a 5×5 normal-equations solve (Cholesky, unrolled) instead of a general 6×6 inverse.
  5. Warp-level argmin over seeds with `__shfl_down_sync` (value + index), then a block-level reduction in shared memory, so the best solution is chosen on GPU without a host round trip.
  6. CUDA Graphs or a persistent kernel to remove per-frame launch overhead (cuRobo reports about 10× from CUDA Graphs in planning).
  7. fp16/`half2`: Turing has fast fp16 (spec, not cited), but IK needs about 0.1 mm precision over a roughly 0.4 m reach, about 4 significant digits, which is borderline for fp16. Present it as an **experiment that shows a precision/speed trade-off** (e.g. fp16 for the coarse seed-screening stage, fp32 polish, the same two-stage idea as HJCD-IK), not as the production path.
  8. Alternative layout comparison: one thread per seed vs HJCD-IK's one block per seed. For 5 DOF, one thread per seed is almost certainly better because there is too little per-seed work to split.

### Gaps
- No published batched-IK timings for a 5-DOF arm or for an RTX 2060 / sm_75. All RTX 2060 numbers above are extrapolations.
- No published seeds/sec for pytorch_kinematics or PyRoki in isolation (PyRoki appears only in HJCD-IK's table).
- No source quantifies how often a hand-teleop IK flips solutions with vs without multi-seed continuity selection. This has to be shown empirically (e.g. count of frames where ‖Δq‖ exceeds a threshold).

---

## Q2. Sampling-based MPC (MPPI, STORM, cuRobo MPC, pytorch_mppi): samples × horizon × Hz, and latency anticipation

### Takeaway
STORM ran joint-space MPPI on a 7-DOF Franka at **125 Hz (<8 ms per command)** on a Titan RTX with **500 particles × 30-step horizon** in PyTorch. cuRobo's MPPI reached **up to 421 Hz** on an RTX 4090, 3.36× STORM. For a 5-DOF arm at 50 Hz, MPPI with about 500–2000 samples × 20–30 steps is comfortably feasible on an RTX 2060. This is a real, visible improvement: smooth, jerk-limited motion that respects the STS3215 velocity/acceleration limits, plus a natural place to fold in a *predicted* hand trajectory. A custom fused rollout+cost kernel is a strong, non-artificial GPU showcase.

### Cited Findings
- STORM: joint-space sampling-based MPC (MPPI-style, covariance updated), Franka Panda, <8 ms per command (125 Hz) on one GPU. It claims about 100× speedup over prior approaches. It was implemented in **PyTorch**, ran MPC in a separate process, and the timing table is on a **Titan RTX**. — [STORM, arXiv 2104.13542](https://arxiv.org/abs/2104.13542) (paper PDF, Appendix A)
- STORM key design: the whole batch × horizon of states is computed **with one tensor op** (closed-form integration of sampled accelerations), with no sequential loop over the horizon, "key to achieving the 8ms control latency". It uses variable dt (smaller near the start of the horizon). — [STORM PDF](https://arxiv.org/pdf/2104.13542)
- STORM ablations used 100–500 particles. Timing figure 14c uses "500 particles with horizon of 30 steps", and latency stays roughly constant as the particle count increases. Smooth, low-jerk motion was achieved "even with 200 particles". Halton sampling + B-spline smoothing of samples, and "null" (zero-acceleration) particles, help smoothness and holding still. The GPU forward model is about 5× faster than the CPU baseline at 500 particles. — [STORM PDF](https://arxiv.org/pdf/2104.13542)
- STORM self-collision: a learned MLP (jointNERF, [256,128,64], sin/cos encoding of q) predicts minimum link distance. It is over 40× faster than FK+mesh-distance and holds 0.4–0.6 ms even at large batches. Environment collision used a learned point-cloud classifier. — [STORM PDF](https://arxiv.org/pdf/2104.13542)
- STORM's related-work comparison: operational-space controllers achieve 1–2 ms latency, gradient-based joint MPC 20–125 ms, and GPU MPPI for off-road driving 40 Hz with 1200 samples and a 2.5 s horizon. — [STORM PDF](https://arxiv.org/pdf/2104.13542)
- cuRobo implements MPPI, gradient descent and L-BFGS in one framework. MPPI reaches up to 421 Hz, 3.36× STORM (RTX 4090). Trajectory optimization uses 32 timesteps. Particle optimization (2 iterations) initializes L-BFGS, adding about 2 ms. The authors list reactive (fixed-rate) generation as a limitation. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- pytorch_mppi (UM-ARM-Lab, MIT): batched MPPI, SMPPI (smoothed in action-derivative space; "doesn't work well" on their toy problem) and KMPPI (kernel/B-spline-interpolated control points, `num_support_pts`; "works well" for smoothness). Parameters: `num_samples`, `horizon`, `lambda_` (try 1e-2), `noise_mu`, `terminal_state_cost`. Pass `device` for GPU. "Runtime scales better with num_samples than horizon", so increase samples first. No timings are published. — [pytorch_mppi GitHub](https://github.com/UM-ARM-Lab/pytorch_mppi)
- Prior art on predicting operator motion to compensate teleop delay exists, but outside MPPI: a grey-predictor on a PUMA arm reduced forward latency from about 350 ms to 250 ms, model-mediated prediction appears in shared-control teleop, and commands are sampled and predicted to offset motion-to-photon latency. I found **no paper combining MPPI with hand-motion prediction for teleop latency**. — [Grey predictor (ITU)](https://research.itu.edu.tr/en/publications/predictive-input-delay-compensation-with-grey-predictor-for-netwo/); [IEEE 10394188](https://ieeexplore.ieee.org/document/10394188); [arXiv 2407.16591](https://arxiv.org/pdf/2407.16591); [arXiv 2608.13284](https://arxiv.org/pdf/2608.13284)

### Inferences
- **What MPPI buys for this arm:** the current chain (EMA → IK → position command) has no notion of servo dynamics. MPPI over joint-acceleration (or jerk) samples with costs `‖FK(q_h) − x̂_hand(t+h)‖² + w·‖q̈‖² + w·jerk + limit/velocity penalties (≤ ~3000 ticks/s, about 4.6 rad/s) + collision` yields:
  - **smoothness** (sampled B-spline/low-pass controls, as in STORM and KMPPI),
  - **velocity/acceleration limits respected by construction**, so the servos don't saturate and lag,
  - **latency hiding**: if the hand target over the horizon is a *prediction* (constant-velocity or Kalman extrapolation of MediaPipe landmarks), the controller starts moving toward where the hand will be. The total pipeline latency (camera, MediaPipe, ROS hops, 50 Hz bus) is plausibly tens of ms to over 100 ms (uncited; measure it), so predicting about 50–100 ms ahead is the relevant regime. Caveat: prediction overshoots when the hand stops abruptly. Mitigate with null/zero-velocity particles (STORM) and a short prediction horizon.
  - **branch consistency**: MPPI in joint space is local (warm-started from the previous mean), so it naturally avoids IK branch flips. Pair it with batched IK (Q1) to seed/reset it when tracking is lost.
- **Sizing on the RTX 2060 (extrapolated):** 5 DOF instead of 7, and 50 Hz needed instead of 125 Hz, give about 3–4× more slack than STORM had on a Titan RTX (Titan RTX ≈ 16 TFLOPS fp32 vs RTX 2060 ≈ 6.5 TFLOPS: spec numbers, uncited). Budget: 1024–4096 samples × 30 steps × 5 joints = 150k–600k FK evaluations per MPC iteration, which is trivial for a fused kernel (cuRobo: about 1 ns/FK at large batch on a 4090). Memory: 4096 × 30 × 5 × 4 B ≈ 2.5 MB for controls, plus a similar amount for states. That fits easily in 6 GB.
- **Custom kernel shape:** one thread per (sample) running the whole horizon sequentially in registers (5 joints × state), or one thread per (sample, timestep) with the state precomputed by a prefix sum (STORM-style closed form). It accumulates the cost and writes one float per sample. Then comes a **softmax-weighted reduction** (MPPI update: w_k ∝ exp(−(S_k − min S)/λ); weighted mean of noise), done with warp shuffles for min and sum. On-GPU noise via cuRAND (Philox) or a precomputed Halton table. Steps you can show: PyTorch baseline (pytorch_mppi), fused rollout+cost kernel, fused reduction, CUDA Graph capture of the whole iteration.
- **Risk:** MPPI tuning (λ, noise σ, cost weights) is the main time sink in 36 h, and pytorch_mppi's README itself recommends autotuning. Keep the existing IK path as a fallback and gate MPPI behind a flag.

### Gaps
- No published MPPI numbers on an RTX 2060, on a 5-DOF arm, or for hand teleoperation specifically.
- STORM's exact per-component ms breakdown (the Table 1 values) was not extracted from the PDF text, only the 8 ms / 125 Hz total and the figure captions.
- cuRobo MPPI's particle count and horizon behind the 421 Hz figure were not in the extracted text.
- No source on MediaPipe-to-servo end-to-end latency for comparable hobby setups; this must be measured.

---

## Q3. Collision checking (sphere/SDF/ESDF/nvblox, self-collision, table/obstacles): cost per check, batching

### Takeaway
GPU collision checking is extremely cheap at scale. cuRobo reports about 10 ns per collision query at batch 100k and distance queries up to 16,000× faster than PyBullet. STORM's learned self-collision runs in 0.4–0.6 ms for large batches. For a 5-DOF arm next to a table, sphere-vs-plane/box costs are a few dozen flops per sphere, and including them inside the IK/MPPI kernel costs almost nothing. The visible benefit is real: the arm never drives the gripper into the table or itself when the hand dips too low. A full ESDF/nvblox camera map is overkill for 36 h.

### Cited Findings
- cuRobo: collision checking about 10 ns/query at batch 100k. Distance queries up to 16,000× faster than PyBullet at 100k. World collision activation distance 2.5 cm, plus continuous (swept) checking. Only about 50% of sphere pairs need to be in the self-collision pair set. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- cuRobo integrates nvblox ESDF built from a wrist-mounted RealSense D-415 on a UR10e. cuRobo v2 advertises "GPU-native ESDF perception … up to 10x faster than state-of-the-art" and "map-reduce self-collision". — [cuRobo paper](https://arxiv.org/html/2310.17274v2); [cuRobo README](https://raw.githubusercontent.com/NVlabs/curobo/main/README.md)
- STORM: a learned self-collision MLP is 40× faster than FK + link-distance, at 0.4–0.6 ms for large batches. — [STORM PDF](https://arxiv.org/pdf/2104.13542)
- PyRoki supports sphere, capsule, halfspace and heightmap collision primitives (meshes approximated as capsules) and self/world collision costs. — [PyRoki GitHub](https://github.com/chungmin99/pyroki)
- AnyTeleop's collision avoidance is built on CUDA-based geometry queries (secondary news source). — [Hackster](https://www.hackster.io/news/taking-matters-into-your-own-hands-51855b74b5b4)

### Inferences
- For the SO-101, about 10–20 spheres (hand-placed along links and gripper) plus a table halfspace and 1–3 boxes/spheres for obstacles is enough. Per configuration that is ≤20 sphere-plane checks and ≤~100 sphere-pair checks for self-collision (adjacent links excluded). This adds roughly a few hundred flops per FK evaluation, so it fuses into the IK/MPPI kernel at near-zero marginal cost. A separate collision workload is **not** heavy enough to justify GPU on its own at this scale. It is a cost term inside Q1/Q2.
- A **visible demo**: put a box on the table, move the hand "through" it, and show the arm going around (MPPI) or stopping at the boundary (IK with a collision penalty) instead of colliding.
- A learned self-collision MLP (STORM) is unnecessary for 5 DOF with few spheres, because explicit spheres are cheaper and exact enough.
- nvblox/ESDF needs a depth camera and extra integration. Skip it in 36 h unless a depth camera is already in the pipeline.

### Gaps
- No published sphere count or per-check cost for SO-ARM101 / SO-100 class arms.
- No cuRobo per-sphere timing on Turing.

---

## Q4. Motion retargeting optimization (dex-retargeting, AnyTeleop, Open-TeleVision, bunny-VisionPro): what compute they use

### Takeaway
Hand-to-robot retargeting systems run their per-frame retargeting optimization on **CPU** (dex-retargeting is Pinocchio-based with small nonlinear optimizers), at camera rate (AnyTeleop is designed for 25 Hz detection/retargeting). GPU is used for collision-aware motion generation (AnyTeleop uses CUDA geometry queries). Per-frame retargeting of a 5-DOF arm is far too small to need a GPU. GPU only helps when you batch it (multi-seed, or over a horizon as in MPPI/trajectory optimization).

### Cited Findings
- dex-retargeting (MIT): kinematics "based on pinocchio". It provides `PositionOptimizer` (from "From One Hand to Multiple Hands"), a vector-style optimizer, and `DexPilotOptimizer`. The README reports no speed figures. — [dex-retargeting GitHub](https://github.com/dexsuite/dex-retargeting)
- AnyTeleop: hand pose detection is designed for at most 25 Hz, and retargeting/fusion run at that rate. Motion generation is designed for 120 Hz, which was hard to reach with all modules on one computer; the authors suggest a separate machine for control. — [AnyTeleop, arXiv 2307.04577 (via search summary; full PDF too large to fetch)](https://arxiv.org/pdf/2307.04577)
- A 2026 dexterous-hand retargeting paper reports 9.05 ms average retargeting latency with linearized collision constraints. — [arXiv 2603.29213](https://arxiv.org/pdf/2603.29213)
- MIRROR (2026): a **GPU-parallelized, continuation-based differential IK** for humanoid teleop from visual skeletal pose. It runs multiple constrained IK QPs in parallel with a self-collision CBF and picks the update that best reduces global task error (Lyapunov-based progression), so it escapes local minima near joint limits, singularities and collision boundaries. The abstract gives no Hz or candidate count. — [MIRROR, arXiv 2603.23995](https://arxiv.org/abs/2603.23995)

### Inferences
- MIRROR is the closest published analogue to the multi-seed-and-select idea in Q1: parallel IK candidates on GPU, choosing the one that makes stable progress. It is a citable justification that "GPU parallel candidates for teleop IK" is a real research direction, not an artificial one.
- For this project, "retargeting" means the hand-to-EE-offset map (scale, deadzone, workspace clamp) plus a choice of what the 5-DOF wrist tracks (pitch/roll vs full orientation). Those are design choices, not compute.
- I did not verify Open-TeleVision and bunny-VisionPro in this session. From prior knowledge (unverified), they use CPU IK (Pinocchio/CasADi or similar) at headset rate, so treat them as a gap.

### Gaps
- AnyTeleop's per-module ms table was not retrievable (PDF over the fetch limit).
- Open-TeleVision and bunny-VisionPro compute details were not researched in this session.

---

## Q5. Other GPU-shaped planning/control work; cuRobo on Turing/WSL2 and licensing; feasibility ranking

### Takeaway
**cuRobo's license changed:** v1 (≤ v0.7.8, 2023) used the NVIDIA non-commercial ("research or evaluation purposes only") license. **cuRoboV2 (initial release 2026-04-18) is Apache-2.0.** Hardware support is murky for an RTX 2060: v1 docs say "Volta or newer, ≥4 GB VRAM" (Turing qualifies), while v2 docs say "NVIDIA GPU > Turing", driver ≥ 580.65.06 and Ubuntu, with only "experimental Windows" support historically and nothing about WSL2. Installing cuRobo in a 36-h hackathon on WSL2 + Turing is a gamble. Writing a small custom kernel for the 5-DOF case is lower-risk and is what makes "custom GPU kernel optimization" presentable. A precomputed GPU reachability/manipulability map is a cheap, visual bonus.

### Cited Findings
- cuRobo v0.7.8 LICENSE, "NVIDIA License" §3.3: "The Work and any derivative works thereof only may be used or intended for use non-commercially … 'non-commercially' means for research or evaluation purposes only." — [cuRobo v0.7.8 LICENSE](https://raw.githubusercontent.com/NVlabs/curobo/v0.7.8/LICENSE)
- cuRobo main branch: "cuRobo is released under the Apache 2.0 license". LICENSE git history: "release repository" (2023-10-26), then "curobov2 initial release" (2026-04-18). v2 "is a significant rewrite and the public API has changed", so pin `v0.7.8` for the v1 API. v2 is built on PyTorch, CUDA and **Warp**. — [cuRobo README](https://raw.githubusercontent.com/NVlabs/curobo/main/README.md); [GitHub API commit history for LICENSE](https://api.github.com/repos/NVlabs/curobo/commits?path=LICENSE)
- cuRobo v1 install docs: "NVIDIA GPU with VOLTA or newer architecture", at least 4 GB VRAM, "experimental support for Windows". They do not mention WSL2. — [curobo.org install](https://curobo.org/get_started/1_install_instructions.html)
- cuRobo v2 install docs: Ubuntu ≥ 20.04, "**NVIDIA GPU > Turing** and at least 4 GB VRAM", driver ≥ 580.65.06 (CUDA 12+), Python ≥ 3.10 (> 3.13 not validated). The extras are `cu12` (torch ≥ 2.5) or `cu13` (torch ≥ 2.9), and kernels are compiled at runtime via `cuda.core`. — [cuRobo v2 installation.rst](https://raw.githubusercontent.com/NVlabs/curobo/main/docs/getting-started/installation.rst); [pyproject.toml](https://raw.githubusercontent.com/NVlabs/curobo/main/pyproject.toml)
- HJCD-IK's benchmarks ran under Windows 11 + WSL Ubuntu 24.04 with CUDA 12.5 on an RTX 4060 Laptop, which shows that custom CUDA IK kernels work fine under WSL2. — [HJCD-IK](https://arxiv.org/html/2510.07514v1)
- Past RTX 2060 + WSL2 CUDA problems (2020–2021) were driver/Insider-build issues and are likely resolved, but I did not verify that. — [NVIDIA forum](https://forums.developer.nvidia.com/t/cant-run-cuda-sample-in-wsl2/156653); [NVIDIA forum, TensorRT on WSL2](https://forums.developer.nvidia.com/t/tensorrt8-wsl2-issues/199155)
- Reachability on GPU: cuRobo demos 500 sampled poses at about 15 Hz for a live reachability visualization. — [cuRobo paper](https://arxiv.org/html/2310.17274v2)
- cuRobo trajectory optimization: 32 timesteps, 12–28 seeds, L-BFGS (history 4) converging in about 100–200 iterations, about 50 ms mean motion generation (30 ms median on PC, about 100 ms on Orin at 60 W). — [cuRobo paper](https://arxiv.org/html/2310.17274v2)

### Inferences
- **"> Turing" is ambiguous.** It could mean "Turing and newer" or "strictly newer than Turing" (Ampere+). Treat sm_75 as **unsupported/untested for cuRobo v2** until someone runs `pytest --pyargs curobo.tests` on the 2060. Fallback: v0.7.8 (Volta+ stated, so Turing should be fine) with the non-commercial license, which is acceptable for a hackathon/research demo. That build compiles CUDA extensions with torch's toolchain, which can eat hours on WSL2.
- Even if cuRobo works, using it makes the "custom kernel" claim weaker. Use it as a **reference/baseline** to compare against if it installs quickly, and do not depend on it.
- **Other GPU-shaped ideas, ranked for this project:**
  1. **Batched multi-seed IK kernel with continuity selection (Q1).** Low risk, about 4–8 h, a directly visible reduction in flips and failure stalls, a clean optimization story. **Top pick.**
  2. **MPPI on joint accelerations with hand-motion prediction (Q2).** Medium risk (tuning), about 10–16 h, a visible improvement in smoothness, latency and limit respect. Strongest "wow" factor. It builds on the kernel from #1 (FK is shared).
  3. **Sphere collision costs (table plus one obstacle) fused into #1/#2 (Q3).** Low marginal cost and a nice demo. Collision is not a standalone heavy workload.
  4. **Reachability/manipulability map:** brute-force FK over a joint grid (e.g. 64⁵ ≈ 1.07 × 10⁹ configs, or a coarser grid), voxelize the EE positions and per-voxel max manipulability with atomics, then render it in RViz/Gazebo. It is used to clamp the hand workspace and to show users where the arm can reach. GPU-shaped (embarrassingly parallel, atomic reductions into a voxel grid) and about 2–4 h. Offline, though, so less "real-time".
  5. **IKFlow-style learned IK:** needs training, has accuracy issues, and is not worth it for 5 DOF in 36 h.
- **WSL2 practicalities (inference):** CuPy (`cupy-cuda12x`) with `RawKernel`/`RawModule` compiles via NVRTC at runtime with no system nvcc needed, which is the least toolchain friction. Numba CUDA also works but gives less low-level control (no explicit `__shfl_*` ergonomics before newer versions; cooperative groups are limited). Triton supports sm_75 only partially: some features (e.g. bf16, certain `tl.dot` paths) need sm_80+, so Triton is a worse fit for small-matrix per-thread math anyway. These are prior-knowledge claims, not verified in this session.
- **Integration latency:** whichever GPU stage is added, keep the CUDA context warm in a long-lived ROS node (the existing `ik_node.py`), avoid per-frame host↔device allocations (preallocate CuPy arrays), and transfer only the 5 output joint values back. The GPU call should then add well under 1 ms (inference) to a 20 ms (50 Hz) budget.

### Gaps
- No primary source confirms cuRobo v2 works or fails on sm_75 or under WSL2. No GitHub issue was found in this session (the GitHub code search API needed authentication).
- Triton/Numba sm_75 limitations were not verified against current docs in this session.
- RTX 2060 spec numbers (6.5 TFLOPS fp32, register file size, fp16 rate) were used from background knowledge and are uncited.
