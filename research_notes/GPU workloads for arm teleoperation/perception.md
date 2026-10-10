# GPU-heavy hand and scene perception workloads for hand → SO-ARM101 teleoperation (RTX 2060 6 GB, 36 h)

Scope note: the RTX 2060 is Turing (sm_75), 6 GB GDDR6 ([PNY](https://www.pny.com/pny-geforce-rtx-2060-6gb-blower)). Most published FPS numbers below were measured on much faster GPUs (RTX 4090, A100, A800, RTX 4060 Ti, 2080 Ti). Each one is flagged. Assume an RTX 2060 delivers roughly 0.5x a 2080 Ti and much less than a 4090. That ratio is my own estimate, not a sourced benchmark.

## Q1. Which upgrades give a visible improvement in a live demo, and which are compute-heavy enough that a GPU kernel clearly matters?

### Takeaway
The best visible improvement for the risk is metric 3D from two cheap webcams: MediaPipe on each view plus calibrated DLT triangulation gives real-world wrist position and hand orientation. That compute is tiny, though: 21 points times 2 views is microseconds of linear algebra, so a custom kernel there is a gimmick. GPU-heavy workloads that clearly need the GPU are MANO mesh recovery (HaMeR/WiLoR, ViT-L/H backbones), model-based particle or PSO fitting, segmentation (SAM variants), dense optical flow, and TSDF/ESDF mapping. Of these, only a few are real custom-kernel targets: particle-likelihood evaluation, voxelization/TSDF integration, and image preprocessing. The neural models are better left to PyTorch, TensorRT, or ONNX Runtime.

### Cited Findings
**Metric 3D hand via stereo or multi-view triangulation**
- Single-view MediaPipe depth is not usable for metric 3D. One multi-view paper notes that MediaPipe can predict 3D hand poses, but the per-joint depth from a single view is not accurate, so it ignores that depth and triangulates 2D joints across views instead. — [V-Hands, arXiv 2409.13347](https://arxiv.org/pdf/2409.13347)
- MediaPipe's 3D landmark information is relative to the other landmarks and does not give accurate real-space location. — [Smart Design "Hand detection in 3D space"](https://smartdesignworldwide.com/?p=8421)
- Prior art for the cheap rig: a project used 2 Logitech C270 webcams plus MediaPipe hand landmarks with stereo vision to estimate point depth. — [namolert/ComputerVision (GitHub)](https://github.com/namolert/ComputerVision). Another describes two fixed mono cameras reconstructing 3D hand points from MediaPipe 2D features with OpenCV. — [JTE paper](https://www.jte.edu.vn/index.php/jte/article/download/1735/1408)
- Standard method: stereo calibration estimates each camera's internal (intrinsic) and external (extrinsic) parameters. You then match 2D landmarks across views and triangulate. — [Amrita publication](https://amrita.edu/?p=318960); [Smart Design](https://smartdesignworldwide.com/?p=8421)
- MediaPipe Hand Landmarker outputs handedness, image-coordinate landmarks, and "world coordinate" landmarks. Its model bundle is a palm detector at 192x192 plus a landmark model at 224x224, float16, trained on about 30K real images plus synthetic renders. — [Google MediaPipe Hand Landmarker docs](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- Published MediaPipe latency is 17.12 ms on CPU and 12.27 ms on GPU for the whole pipeline, measured on a Pixel 6 phone, not a PC. — [MediaPipe docs](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

**MANO mesh recovery (HaMeR, WiLoR, and others)**
- HaMeR ViT-H: 671 M params, 167.3 GFLOPs, 27 FPS on an RTX 4060 Ti, a consumer GPU faster than the 2060. Distilled students: ConvNeXt-L at 240 M params and 40 FPS, ResNet-50 at 69.3 M and 50 FPS, MobileNet-L at 42 M and 48 FPS, all on the same 4060 Ti. The best distilled model is about 1.48x faster with 64% fewer parameters, losing about 0.4 mm accuracy. — [Fast-HaMeR, arXiv 2603.16444](https://arxiv.org/html/2603.16444v1)
- WiLoR detector: Proposed-M runs at 138 FPS (25 MB) and Proposed-S at up to 175 FPS (7 MB), on an **RTX 4090**. It is 45x faster than ContactHands (3 FPS). The reconstruction stage is a ViT-Large backbone initialized from ViTPose, and **the paper reports no reconstruction FPS**. — [WiLoR, arXiv 2409.12259](https://arxiv.org/html/2409.12259)
- WiLoR accuracy is PA-MPJPE 7.5 mm and PA-MPVPE 7.7 mm, as cited in Fast-HaMeR Table 6. — [Fast-HaMeR](https://arxiv.org/html/2603.16444v1)
- WiLoR repo: a `--fast` flag (fp16 plus depth pruning) gives "up to 1.6x faster" inference with ΔMPJPE of about 0.05 mm. It was tested on PyTorch 2.0.0 and CUDA 11.7. It needs MANO_RIGHT.pkl, which requires MANO registration. Models are CC-BY-NC-ND. The pip-installable "WiLoR-mini" is for demos only. — [WiLoR GitHub](https://github.com/rolpotamias/WiLoR)
- OmniHands, a related ViT-H 4D hand model, reports 16.05 FPS on a single **A800** with Python. Its authors say this is below the 25–30 FPS needed for real time. — [OmniHands, arXiv 2405.20330](https://arxiv.org/pdf/2405.20330)
- Lighter real-time mesh methods: a 2024 "simple baseline" reports 33 FPS (HRNet) and 70 FPS (FastViT-MA36) on a 2080 Ti. — [arXiv 2403.01813](https://arxiv.org/html/2403.01813v1). ReJSHand reports about 72 FPS on a 2080 Ti-class GPU. — [arXiv 2503.05995](https://arxiv.org/pdf/2503.05995). MLPHand (multi-view) reports 71 FPS on an RTX 3090. — [arXiv 2406.16137](https://www.arxiv.org/pdf/2406.16137)
- An older single-image hand mesh method ran at about 19.9 ms per frame (about 50 FPS) on a GTX 1080. — [arXiv 1903.00812](https://arxiv.org/pdf/1903.00812)

**Model-based tracking (PSO and particle filters, analysis-by-synthesis)**
- The FORTH tracker (Oikonomidis, Kyriazis, Argyros) uses GPU model-based optimization on Kinect input at about 20 Hz. — [ERCIM News](https://ercim-news.ercim.eu/en95/special/tracking-the-articulated-motion-of-human-hands-in-3d). Other FORTH GPU PSO material reports about 15 Hz. — [hgpu.org](https://hgpu.org/?p=6897). A KTH deliverable reports about 12 Hz for a GPU prototype and describes a 54-D two-hand PSO. — [KTH GRASP D25](https://www.csc.kth.se/grasp/deliverables/D25.pdf)
- A short-baseline stereo RGB tracker runs at 15–20 FPS for a single hand with 32 particles x 32 generations on a GTX 970. The authors note that PSO "is inherently parallelizable since the particles are only synchronized at the end of each generation." — [Back to RGB, arXiv 1705.05301](https://arxiv.org/pdf/1705.05301)
- A 2013 PSO-with-resampling variant reports 40 FPS on a GTX 580. — [Visual Computer 2013](https://unpaywall.org/10.1007%2FS00371-013-0894-1)
- A paper studies feasibility of real-time 3D hand tracking with edge GPGPU acceleration. — [arXiv 1804.11256](https://arxiv.org/pdf/1804.11256)

**Scene perception: segmentation**
- EfficientViT-SAM: 48.9x TensorRT speedup over SAM-ViT-H on an **A100**. L0 runs at 8.2 ms on Jetson AGX Orin and 762 img/s on A100 (batch 16). XL1 runs at 37.2 ms on Orin and 182 img/s on A100, all TensorRT fp16 including data transfer. — [EfficientViT-SAM HF card](https://huggingface.co/mit-han-lab/efficientvit-sam); [arXiv 2402.05008](https://arxiv.org/html/2402.05008v2)
- MobileSAM: about 12 ms per image on "a single GPU" (8 ms encoder, 4 ms decoder). This is a secondary source and the GPU is unspecified. — [Ultralytics MobileSAM doc](https://dagshub.com/Ultralytics/ultralytics/src/profile-model/docs/en/models/mobile-sam.md)
- FastSAM: no GPU or TensorRT latency found. Ultralytics' CPU table lists FastSAM-s at 55.9 ms/img on an M4 Pro. — [Ultralytics doc](https://dagshub.com/Ultralytics/ultralytics/src/profile-model/docs/en/models/mobile-sam.md)

**Scene perception: TSDF/ESDF (nvblox)**
- nvblox is "a GPU-accelerated TSDF and ESDF library for robots equipped with RGB-D cameras," with C++, Python, and ROS 2 interfaces. — [nvidia-isaac/nvblox](https://github.com/nvidia-isaac/nvblox)
- nvblox_torch maps from depth images (examples use Sun3D and RealSense) and answers signed-distance queries. It builds against the PyTorch C++ ABI. The authors call Docker "the most stable way." Its last noted update was Dec 2023 (nvblox 0.0.5), and install points to a personal fork. — [NVlabs/nvblox_torch](https://github.com/NVlabs/nvblox_torch); [cuRobo install](https://curobo.org/get_started/1_install_instructions.html)

**Dense optical flow**
- Turing GPUs (except TU117) have a fixed-function optical flow and stereo disparity engine, exposed by the NVIDIA Optical Flow SDK. — [NVIDIA Optical Flow SDK](https://developer.nvidia.com/optical-flow-sdk); [NVOFA app note](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-application-note). The RTX 2060 is TU106, so it is not the excluded TU117. That chip ID is from my own knowledge, not a cited source.
- RAFT: about 60 ms per flow on a 2080 Ti (with occlusion and uncertainty heads). — [MFT, arXiv 2305.12998](https://arxiv.org/pdf/2305.12998). It takes about 0.125 s per frame at 32 iterations on an RTX 2080. — [NeuFlow v2, arXiv 2408.10161](https://arxiv.org/pdf/2408.10161). NeuFlow targets real-time flow on edge devices. — [NeuFlow, arXiv 2403.10425](https://arxiv.org/pdf/2403.10425)
- Meta used NVIDIA's optical flow engine for ASW (frame extrapolation for latency hiding) in VR. — [Meta dev blog](https://developers.meta.com/vr/blog/asw-and-passthrough-with-nvidia-optical-flow/)

### Inferences
- **Most visible improvement:** stereo triangulation turns "hand moves in image plane" into a metric 3D hand position and gives a real palm normal for wrist pitch and roll. That directly improves the arm mapping. Second most visible: a gripper aperture from the 3D thumb–index distance, which is more scale-invariant than a 2D distance.
- **MANO mesh (WiLoR/HaMeR)** looks impressive as a rendered mesh overlay and gives better occlusion robustness and full 3D wrist rotation. On a 2060 6 GB, ViT-H HaMeR (671 M params) is probably at most about 10–15 FPS. That estimate scales from 27 FPS on a 4060 Ti and is unverified. Without fp16, 6 GB is tight. The WiLoR ViT-L plus `--fast` fp16 path is the most plausible "real-time-ish" option, but no consumer-GPU number exists.
- **Where a custom kernel clearly matters:** (a) particle/PSO likelihood evaluation, with hundreds to thousands of hypotheses x 21 joints x pixels or points, which is embarrassingly parallel; (b) depth → point cloud → voxel hashing or TSDF integration; (c) fused image preprocessing (undistort, resize, normalize, NCHW pack) feeding two camera streams into models. Batched DLT over 21 points or a Kalman filter are too small to benefit; GPU launch overhead would exceed the compute.
- **Obstacle awareness** (nvblox or SAM) is the least visible improvement and the highest risk. It needs a depth camera or a stereo depth pipeline plus camera-to-robot extrinsics. It also adds no value to a teleop demo unless the IK side actually consumes the ESDF.

### Gaps
- No published FPS for WiLoR reconstruction or HaMeR on any RTX 20xx GPU. VRAM usage for HaMeR or WiLoR is not reported anywhere I found.
- No runtime numbers found for FrankMocap, HandOccNet, or MeshGraphormer. MeshGraphormer appears only as an unlabeled point on a 2080 Ti speed/accuracy plot.
- No source for MediaPipe triangulation accuracy in mm for a two-webcam rig.

## Q2. Concrete numbers: FPS on RTX 20xx/30xx-class GPUs, model sizes, VRAM

### Takeaway
Hard numbers on 20xx/30xx cards exist only for lighter mesh models (33–72 FPS on a 2080 Ti) and RAFT (60–125 ms on a 2080/2080 Ti). The flagship models (HaMeR, WiLoR, EfficientViT-SAM) were benchmarked on a 4060 Ti, 4090, A100, or A800, so their RTX 2060 performance has to be extrapolated.

### Cited Findings
| Workload | Number | GPU | Source |
|---|---|---|---|
| HaMeR ViT-H | 27 FPS, 671 M params, 167 GFLOPs | RTX 4060 Ti | [Fast-HaMeR](https://arxiv.org/html/2603.16444v1) |
| Fast-HaMeR ConvNeXt-L / ResNet-50 / MobileNet-L | 40 / 50 / 48 FPS; 240 / 69 / 42 M params | RTX 4060 Ti | [Fast-HaMeR](https://arxiv.org/html/2603.16444v1) |
| WiLoR detector M / S | 138 / 175 FPS; 25 / 7 MB | **RTX 4090** | [WiLoR](https://arxiv.org/html/2409.12259) |
| WiLoR `--fast` | up to 1.6x speedup | unspecified | [WiLoR repo](https://github.com/rolpotamias/WiLoR) |
| OmniHands ViT-H | 16.05 FPS | **A800** | [arXiv 2405.20330](https://arxiv.org/pdf/2405.20330) |
| Simple-baseline mesh (HRNet / FastViT) | 33 / 70 FPS | RTX 2080 Ti | [arXiv 2403.01813](https://arxiv.org/html/2403.01813v1) |
| ReJSHand | ~72 FPS | 2080 Ti-class | [arXiv 2503.05995](https://arxiv.org/pdf/2503.05995) |
| MLPHand (multi-view) | 71 FPS | RTX 3090 | [arXiv 2406.16137](https://www.arxiv.org/pdf/2406.16137) |
| MediaPipe Hands full pipeline | 17.1 ms CPU / 12.3 ms GPU | Pixel 6 | [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) |
| EfficientViT-SAM L0 / XL1 | 8.2 / 37.2 ms (Orin); 762 / 182 img/s (A100, bs16) | Orin / **A100** | [HF card](https://huggingface.co/mit-han-lab/efficientvit-sam) |
| MobileSAM | ~12 ms (8 + 4) | unspecified GPU | [Ultralytics](https://dagshub.com/Ultralytics/ultralytics/src/profile-model/docs/en/models/mobile-sam.md) |
| RAFT | ~60 ms/flow | RTX 2080 Ti | [MFT](https://arxiv.org/pdf/2305.12998) |
| RAFT (32 iters) | ~125 ms/frame | RTX 2080 | [NeuFlow v2](https://arxiv.org/pdf/2408.10161) |
| PSO stereo hand tracker | 15–20 FPS (32 particles x 32 generations) | GTX 970 | [arXiv 1705.05301](https://arxiv.org/pdf/1705.05301) |
| FORTH PSO (Kinect) | ~20 Hz (other reports 12–15 Hz) | era GPUs | [ERCIM](https://ercim-news.ercim.eu/en95/special/tracking-the-articulated-motion-of-human-hands-in-3d); [hgpu](https://hgpu.org/?p=6897) |

### Inferences
- Jetson AGX Orin is roughly in the same performance class as a 2060 for fp16 TensorRT inference, but that comparison is my own rough heuristic and unverified. If it holds, EfficientViT-SAM L0 at about 8 ms suggests near-real-time segmentation on a 2060 with TensorRT. In plain PyTorch it would likely be several times slower.
- A 2060's fp16 tensor cores make fp16 or TensorRT essential for ViT-L/H models. In fp32, ViT-H HaMeR weights alone would be about 2.7 GB (671 M x 4 B), my own arithmetic, which leaves little headroom in 6 GB alongside MediaPipe, a desktop, and other processes.
- RAFT at 60–125 ms on a 2080-class card would be about 4–8 FPS on a 2060 (extrapolated). That is too slow for latency reduction. The hardware NVOF engine is the only real-time dense-flow path on this card.

### Gaps
- No measured VRAM for any of these models. No RTX 2060 measurements found for any model.
- No per-frame NVOF latency figure for Turing was found in the retrieved material.

## Q3. Good custom-kernel targets vs better left to libraries

### Takeaway
Write custom CUDA/CuPy/Triton kernels for parallel geometric or likelihood work with simple structure: many-particle hand-model likelihoods, depth-to-point-cloud plus voxelization or TSDF fusion, and fused pre-processing. Leave neural backbones (MediaPipe, WiLoR/HaMeR, SAM, RAFT) to TensorRT, ONNX Runtime, or PyTorch, and leave mapping to nvblox if a depth camera exists.

### Cited Findings
- PSO and particle-based hand tracking is "inherently parallelizable since the particles are only synchronized at the end of each generation." The known GPU implementations reach 15–40 FPS with small budgets (32x32), which shows the likelihood evaluation is the bottleneck. — [arXiv 1705.05301](https://arxiv.org/pdf/1705.05301); [Visual Computer 2013](https://unpaywall.org/10.1007%2FS00371-013-0894-1); [ERCIM](https://ercim-news.ercim.eu/en95/special/tracking-the-articulated-motion-of-human-hands-in-3d)
- nvblox already provides optimized GPU TSDF/ESDF with ROS 2 integration, so a hand-written TSDF would duplicate it. — [nvblox](https://github.com/nvidia-isaac/nvblox)
- EfficientViT-SAM's headline 48.9x speedup comes from architecture plus TensorRT, not custom kernels. — [arXiv 2402.05008](https://arxiv.org/html/2402.05008v2)
- WiLoR's speedup path is fp16 plus depth pruning (`--fast`), which is library-level. — [WiLoR repo](https://github.com/rolpotamias/WiLoR)
- Dense flow has a dedicated hardware engine on Turing that a CUDA kernel cannot beat at its own task. — [NVIDIA OF SDK](https://developer.nvidia.com/optical-flow-sdk)

### Inferences
- **Best showcase kernel for this project:** a GPU particle filter over hand pose or wrist pose, with N = 1k–100k particles. Each particle is a candidate 3D hand (or wrist pose plus aperture), and its likelihood comes from reprojection error against both cameras' MediaPipe 2D landmarks. The cost is O(N x 21 x 2 views) projections, which suits one thread per particle, a warp reduction for weights, and parallel systematic resampling. It is measurable (CPU NumPy vs CuPy RawKernel vs Triton), smooths jitter, and handles single-view occlusion by falling back to one-view likelihood. It doesn't need a renderer, which avoids the much heavier analysis-by-synthesis depth-rendering approach of FORTH-style trackers.
- **Second target:** fused undistort, resize, normalize, and NCHW pack for two 720p streams in one kernel. This saves CPU time if the networks also run on GPU, but the gain is small relative to MediaPipe's CPU inference unless the whole pipeline moves to GPU.
- **Third target, only with a depth source:** depth → point cloud → voxel occupancy grid in one CUDA kernel, with atomics into a dense grid sized to the arm workspace (for example, 64^3 at 1 cm). Simple, visual (RViz), and fast. Use it only if a RealSense or OAK-D is available, or if stereo disparity is computed via NVOF or OpenCV.
- **Not worth a kernel:** batched DLT for 21 points (two 4x4 SVDs per point; trivial on CPU), a Kalman filter on 21x3 state, and IK for 5 DOF. A "batched IK over many seeds" kernel could be a credible secondary target, but that falls outside this perception scope.

### Gaps
- Found no published benchmark comparing CuPy, Triton, and CUDA for particle-filter likelihoods specifically.

## Q4. Hardware the team may not have; cheapest viable setup; platform risks (WSL2/Windows)

### Takeaway
The cheapest high-value setup is two USB webcams, rigidly mounted and calibrated once with an OpenCV checkerboard. No depth camera is needed for metric 3D hand pose. Depth-dependent workloads (nvblox, voxel obstacle maps) should be cut unless a RealSense or OAK-D is on hand. The Python MediaPipe GPU delegate on Windows is unproven; plan for CPU MediaPipe with GPU work handled separately.

### Cited Findings
- Two Logitech C270 webcams plus MediaPipe plus stereo triangulation has been done by others. — [namolert/ComputerVision](https://github.com/namolert/ComputerVision)
- Multi-view systems use MediaPipe 2D per view, then multi-view reconstruction, discarding MediaPipe's per-view depth. — [arXiv 2409.13347](https://arxiv.org/pdf/2409.13347)
- nvblox and nvblox_torch are built for RGB-D depth input; examples use RealSense. Docker on Ubuntu is the recommended path, and there is no mention of Windows. — [nvblox](https://github.com/nvidia-isaac/nvblox); [nvblox_torch](https://github.com/NVlabs/nvblox_torch)
- WiLoR was tested on CUDA 11.7 and PyTorch 2.0 and depends on Ultralytics plus MANO (registration required). — [WiLoR repo](https://github.com/rolpotamias/WiLoR)
- MediaPipe GPU on desktop is shaky. A Windows 10 user seeking GPU MediaPipe hands hit dependency conflicts (tensorflow-gpu 2.10 protobuf vs mediapipe ≥0.10.5). — [mediapipe#5897](https://github.com/google-ai-edge/mediapipe/issues/5897). A macOS Python user saw no FPS gain from the GPU delegate. — [mediapipe#6041](https://github.com/google-ai-edge/mediapipe/issues/6041). A Windows-specific Holistic crash has also been reported. — [mediapipe issues](https://github.com/google-ai-edge/mediapipe/issues/5568). A Dart/LiteRT wrapper of the same models lists Windows and Linux as CPU-only on its default interpreter path. — [pub.dev hand_detection](https://pub.dev/documentation/hand_detection/3.3.0/)
- On Jetson, TFLite refused the GPU delegate for MediaPipe hands, capping it to CPU. — [NVIDIA forum](https://forums.developer.nvidia.com/t/hand-tracking-on-jetson-orin-nano/297754)

### Inferences
- **Recommended 36 h architecture:** run capture plus MediaPipe CPU on native Windows (two processes or threads, one per camera; MediaPipe at about 10–20 ms per frame on CPU keeps 30 FPS feasible). Send 2D landmarks over UDP or ZeroMQ to WSL2. Do triangulation, the CuPy particle filter or smoother, and IK in WSL2, where CUDA works via the WSL GPU driver. This avoids passing webcams through to WSL2.
- Camera sync: two unsynchronized USB webcams at 30 FPS can be up to about 33 ms apart. Fast hand motion will bias the triangulation. A particle filter or Kalman filter with timestamps mitigates this. This is my own inference, with no measured error source found.
- **Risk ranking (36 h):** stereo triangulation plus filtering is low risk. A CuPy/Triton particle filter on landmark reprojection is low to medium risk. WiLoR/HaMeR mesh at interactive FPS on a 2060 is medium-high risk (MANO license, CUDA 11.7 pins, unknown FPS, 6 GB VRAM). EfficientViT-SAM via TensorRT for object masks is medium-high risk. nvblox plus obstacle-aware IK is high risk (needs a depth camera, Docker, extrinsics, and planner integration). RAFT-based flow prediction is high risk and too slow; NVOF via OpenCV CUDA is medium risk.

### Gaps
- No primary source found stating definitively whether the MediaPipe Python Hand Landmarker GPU delegate works on Windows.
- No primary nvblox requirements page could be retrieved (the docs URL redirected or returned 404), so minimum compute capability and Turing support were not confirmed from the docs. Turing support is plausible because nvblox targets Jetson Orin and x86 RTX GPUs, but that is unverified here.
- Current prices of RealSense D435 or OAK-D were not researched.
