# Filter baseline (2026-10-10, before Phase 2)

Recordings: `~/kh/rec_{still,sweep,reach,drop_out}.npz` (all with full range calibration).
Logs here were made with the code as of Phase 1 (one-euro 1 Hz / beta 2, `--max-lin 0.25`):

    .venv/bin/python Computer-Vision/common/hand_target_node.py --replay rec_<name>.npz --no-ros --log <name>.npz
    .venv/bin/python Computer-Vision/common/filter_eval.py <name>.npz --window <start> <end>

Scoring windows (s from the first frame; calibration ends ~24-29 s):
still 24.5 38.5 | sweep 35.5 53 (edge-free; 35.5 61.5 full) | reach 25.1 58.6 | drop_out 28.5 47

|                      | still | sweep | reach | drop_out |
|---|---|---|---|---|
| tracking frames      | 420   | 528 | 794   | 286  |
| no-hand share        | 0%    | 0% | 22%   | 49%  |
| jitter raw (mm)      | 1.12  | 4.20 | 2.70  | 3.51 |
| jitter out (mm)      | 0.22  | 1.52 | 1.14  | 1.53 |
| jitter raw (deg)     | 4.46  | 4.93 | 7.69  | 9.48 |
| jitter out (deg)     | 0.51  | 0.92 | 1.82  | 2.72 |
| lag (ms)             | -     | 85 | 90    | 83   |
| accel RMS (m/s2)     | 0.30  | 1.36 | 1.04  | 1.25 |
| accel p99 (m/s2)     | 0.81  | 4.91 | 3.91  | 4.26 |
| at speed cap         | 0%    | 1% | 2%    | 7%   |
| dt std / max (ms)    | 7.9 / 64 | 7.7 / 65 | 10.4 / 127 | 10.5 / 94 |

Notes:
- Sweep re-recorded 12:11 with smaller motions: box edge 0% in 35.5-53 s (53-58 s pushes
  forward to the +10 cm x limit, so the scoring window stops at 53).
- The reach still sits at the offset box edge 29% of the time. Clipped axes are flat,
  which flatters jitter and hides lag there.
- Clutch re-acquire: 0.0 mm output step in all 10 returns (reach + drop_out).
- Still hand: raw noise 1.1 mm / 4.5 deg; rotation is the noisier part.

## Phase 2 result (Kalman default, `recordings/phase2/`)

|                      | still | sweep | reach | drop_out |
|---|---|---|---|---|
| jitter out (mm)      | 0.22  | 1.18  | 1.24  | 1.75 |
| lag (ms)             | -     | 20    | 38    | 41   |
| accel RMS (m/s2)     |       | 0.87  |       |      |
| accel p99 (m/s2)     |       |       | 3.88  |      |
| overshoot mean / max (mm) |  |       | 2.4 / 4.0 (one-euro 1.4 / 2.8) | |

## Phase 2b: IMM (default since, `recordings/phase2_imm/`)

Full pipeline (filter_eval), vs the single Kalman above:

|                      | still | sweep | reach | drop_out |
|---|---|---|---|---|
| jitter out (mm)      | 0.24  | 1.22  | 1.30  | 1.77 |
| lag (ms)             | -     | 15    | 34    | 39   |
| accel RMS (m/s2)     |       | 0.90  |       |      |
| overshoot mean / max (mm) |  |       | 2.3 / 4.0 | |

`--imm responsive`: lag 15 / 21 / 34 ms, overshoot 1.8 / 4.1 mm, sweep accel 1.17, jitter 0.31 / 1.44 / 1.41 / 1.80.
Filter-only scores vs a zero-lag reference: `.venv/bin/python Computer-Vision/common/kalman_bench.py`.
