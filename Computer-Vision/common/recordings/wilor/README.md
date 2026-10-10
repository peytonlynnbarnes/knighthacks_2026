# WiLoR recordings (2026-10-10)

Frames: `~/kh/{still,sweep,reach,drop-out,edge-on}/` (`--record-frames`, recorded live with
`--backend wilor`, so 11–16 pairs/s: the rate WiLoR runs at live). Not in git (~2 MB/s).

- `lm_wl_<take>.npz` / `lm_mp_<take>.npz`: landmarks from each backend on those frames (`--record`),
  `extract_*.txt` their console output. `--replay lm_wl_<take>.npz` reruns the pipeline without the GPU.
- `wl/<take>.npz`, `mp/<take>.npz`: `--log` files, made with the MediaPipe filter presets (before
  the WiLoR tuning). `windows.json` (same in both): the scoring window per take, in s.
- `tune_*.json`: grid searches (scratch script, not kept); `ref` = settings compared against,
  `all` = every setting with its `kalman_bench` scores. `tune_edge_smooth.json` holds the chosen ones.

Caveats: MediaPipe sees only WiLoR's ~15 fps here (a handicap for it). In drop_out the hand
calibration never finished (both backends), so its window uses every frame with a hand pose.

## Measured noise (WiLoR, `kalman_bench --ref-s 0.25`)

| speed m/s | along-ray mm | sideways mm |
|---|---|---|
| still (< 0.03) | 2.35 (MediaPipe 1.43) | 0.85 (MediaPipe 0.35) |
| 0.03–0.20 | 3.4–3.5 (MediaPipe 3.1–3.6) | 0.95–1.08 (MediaPipe 0.69–0.82) |

So WiLoR is noisier mainly on a still hand, and its noise barely grows with speed.

## Filter tuning: `hand_mapping.IMM_PRESETS["wilor"]`, `KALMAN_DEFAULTS["wilor"]`

`kalman_bench.py --logs recordings/wilor/wl --backend wilor`, all 5 takes. Score = mean ratio vs
the MediaPipe smooth preset of each moving take's error + still wobble; no lag > +20 ms allowed.
~2000 IMM and 77 single-Kalman settings.

| filter (on WiLoR data) | still wob | sweep err / lag | reach err / lag / ov | drop err | edge err | reach acc |
|---|---|---|---|---|---|---|
| MediaPipe imm smooth | 0.56 | 5.40 / 20 | 6.32 / 41 / 7.2 | 10.06 | 4.09 | 0.64 |
| MediaPipe imm responsive | 0.66 | 3.69 / 10 | 4.25 / 28 / 5.6 | 6.45 | 3.48 | 0.75 |
| **wilor smooth** (1.5 / 1.5 / 0 / 0.01 / 0.995) | **0.31** | 2.74 / 5 | 2.71 / 12 / 4.6 | 3.33 | 2.85 | 1.03 |
| wilor responsive (1.0 / 1.0 / 0 / 0.01 / 0.99) | 0.40 | 2.83 / 0 | 2.50 / 10 / 4.7 | 3.00 | 2.77 | 1.17 |
| wilor kalman (sigma 1, q 0.0001) | 0.58 | 3.19 / 10 | 3.59 / 15 / 5.5 | 4.38 | 3.32 | 1.13 |

(IMM: sigma_px / ray_px / speed_k / moving q / p_stay. mm, ms, m/s².) The ranking holds against
the tighter reference (`--ref-s 0.135`). Trade-off: ~1.6× the acceleration while moving
(it follows the hand closer); downstream `ik_node` interpolation and the `feetech_mirror` EMA
smooth that further. The best settings lie on a flat ridge (p_stay 0.99–0.999, sigma ≈ ray 1–1.5);
p_stay 0.999 scored ~0.002 better but was not used, as too sticky.

Full pipeline (`--replay lm_wl_<take>.npz`, `filter_eval.py --window`), old → new (wilor smooth):
still jitter out 0.53 → 0.22 mm (MediaPipe on its own data: 0.22); sweep lag 30 → 10 ms,
jitter 2.17 → 2.51; reach lag 42 → 16 ms, jitter 2.05 → 2.46, overshoot 3.6 → 3.5 mm.
MediaPipe replays are bit-identical to before (`rec_reach.npz`).
