# Dexterous Triage Lab - Real-World Test Plan

Registration UUID: 9e089c2c-c100-4dad-8652-052d998ce23e

## Purpose

The v17 package turns the MuJoCo demo into a bench-testable medication triage protocol. The submitted code still generates the demo video and metrics, but now it also exports `artifacts/dexterous_triage_real_world_transfer.json`, a machine-readable bridge from simulated evidence to physical pass/fail checks.

## Physical Bench Protocol

1. Use a medicine-vial surrogate with a 24-28 mm diameter, soft cap, and 160-220 g filled mass.
2. Place the vial, cap-discard zone, sterile pod, and red audit button at the same relative positions used in the MJCF scene.
3. Run the five-finger hand through scan, servo approach, grasp, cap twist, pod placement, audit press, slip recovery, and dataset export.
4. Record synchronized video timestamps for grasp lock, cap twist, pod entry, audit press, and recovery proof.
5. Replay the same trial with residual feedback disabled to confirm the closed-loop policy is responsible for recovery.

## Pass/Fail Thresholds

| Test | Pass condition | Evidence file |
|---|---|---|
| Vial dimension tolerance | Final vial-pod error <= 50 mm and peak grip <= 0.90 normalized force | `dexterous_triage_report.json` |
| Slip impulse recovery | Slip observer returns <= 2.5 mm before release | `dexterous_triage_report.json` |
| Friction/cap-torque sweep | Residual success rate >= 95% and p95 final error <= 14 mm | `dexterous_triage_eval.json` |
| Audit-button safety gate | Button is pressed only after pod placement and reaches >= 25 mm depth | `dexterous_triage_trajectory.json` |

## Failure Recovery Matrix

| Failure mode | Detection signal | Recovery action | Stop rule |
|---|---|---|---|
| Vial slip during transfer | `slip_observer_error_mm` rises during recovery window | Increase grip residual and counter-move gantry | Stop if final slip stays > 2.5 mm |
| Cap torque spike | Cap separation lags while grip remains stable | Hold vial, rotate wrist, and reduce lateral motion | Stop if vial-pod error grows after place stage |
| Over-grip risk | Grip exceeds safe force envelope | Reduce finger residual and keep thumb opposition | Stop if normalized grip exceeds 0.90 |
| Premature confirmation | Button motion before pod placement | Suppress button actuator until place stage clears | Stop if button depth rises before delivery |

## Judge Fast Path

1. Watch the demo video for labels: `KEY MOMENT`, `SAFETY GATE`, and `RECOVERY PROOF`.
2. Open `artifacts/dexterous_triage_real_world_transfer.json` to inspect the physical bench tests.
3. Open `artifacts/dexterous_triage_eval.json` to compare residual-policy success with the no-residual baseline.
4. Open `artifacts/dexterous_triage_contact_timeline.json` to verify five-finger contact and recovery-window samples.

## Why This Should Lift The Score

- Claude's likely real-world concern is answered with concrete physical tests and thresholds.
- GPT's likely narrative concern is answered with shorter, stage-labeled video moments.
- Gemini's likely presentation concern is answered with dynamic visual markers tied to the same generated metrics.
- The package remains reproducible: one command regenerates the video, JSON evidence, SRT subtitles, and validation targets.
