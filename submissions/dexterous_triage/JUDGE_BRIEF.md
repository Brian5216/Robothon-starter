# Dexterous Triage Lab - Judge Brief

Registration UUID: 9e089c2c-c100-4dad-8652-052d998ce23e

## Why This Entry Should Compete For #1

Dexterous Triage Lab is a five-finger, 22-channel MuJoCo dexterous-hand system for a long-horizon medication triage task. It is designed to answer the previous review concern directly: the current version is not only a staged video, it logs closed-loop residual control, raw-vs-corrected visual servo error, contact balancing, a five-finger contact timeline, slip recovery, concise subtitles, dynamic route/residual/slip visuals, and fixed-seed stress-test results.

## What To Inspect First

1. `artifacts/dexterous_triage_demo.mp4` - generated demo video.
2. `artifacts/dexterous_triage_report.json` - success criteria, closed-loop metrics, self-audit.
3. `artifacts/dexterous_triage_eval.json` - 32-rollout stress test and baseline comparison.
4. `artifacts/dexterous_triage_policy_card.json` - policy inputs, outputs, topology, and evidence.
5. `artifacts/dexterous_triage_narration.srt` - stage narration matching the video overlays.
6. `artifacts/dexterous_triage_contact_timeline.json` - five-finger active-contact, balance, stable-hold, and recovery evidence.
7. `dexterous_triage_scene.xml` - five-finger MJCF hand, actuators, sensors, free bodies, and task objects.

## Quantitative Evidence

- Final task completion: 1.0
- Actuated channels: 22
- Hand topology: five fingers - thumb, index, middle, ring, little
- Residual corrections applied in the demo: 1027
- Raw median visual-servo error: 0.02141 m
- Post-residual median visual-servo error: 0.00672 m
- Visual-servo error reduction: 68.61%
- Final slip observer error: 1.079 mm
- Contact timeline: five active fingers, stable contact window, and recovery-window samples
- Stress-test rollouts: 32 fixed seeds
- No-residual baseline success: 68.75%
- Residual-policy success: 100%
- Median final error improvement: 56.386 mm to 10.462 mm

## Rubric Mapping

- Runnability: one command regenerates video, trajectory, report, policy card, and stress evaluation.
- MuJoCo depth: MJCF bodies, free joints, hinge/slide joints, position actuators, frame sensors, touch sensors, contacts, and task zones.
- Task design: long-horizon triage sequence with scan, grasp, uncap, place, confirm, recover, and export phases.
- Control: closed-loop residual controller using visual-servo, contact-target, five-finger contact timeline, and slip-observer feedback.
- Dexterous manipulation: five-finger hand, thumb opposition, cap rotation, fragile vial handling, and placement.
- Engineering quality: deterministic run, validation script, structured artifacts, and fixed-seed ablation.
- Presentation: generated video overlays concise stage text, route trails, scan beams, residual arrows, slip ripples, servo error, slip observer, residual norm, and confidence; SRT subtitles and contact timeline evidence are included.
- Innovation: combines dexterous medication handling, safety confirmation, residual recovery, and dataset export.

## Honest Scope

The high-level task sequence is deterministic for reproducible judging, while the residual policy is closed-loop and stress-tested against disturbances. This balances reliability with measurable control behavior.
