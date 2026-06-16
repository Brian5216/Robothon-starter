from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def fail(message: str) -> int:
    print(f"[fail] {message}")
    return 1


def main() -> int:
    registration_path = ROOT / "registration.json"
    pr_path = ROOT / "PR_DESCRIPTION.md"
    video_path = ROOT / "artifacts" / "dexterous_triage_demo.mp4"
    report_path = ROOT / "artifacts" / "dexterous_triage_report.json"
    trajectory_path = ROOT / "artifacts" / "dexterous_triage_trajectory.json"
    policy_card_path = ROOT / "artifacts" / "dexterous_triage_policy_card.json"
    eval_path = ROOT / "artifacts" / "dexterous_triage_eval.json"
    narration_path = ROOT / "artifacts" / "dexterous_triage_narration.srt"
    judge_brief_path = ROOT / "JUDGE_BRIEF.md"
    scorecard_path = ROOT / "rubric_scorecard.json"
    manifest_path = ROOT / "submission_manifest.json"

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    uuid = registration.get("uuid", "")
    if not UUID_RE.match(uuid):
        return fail("registration.json still needs a valid Robothon UUID.")
    if "PASTE-" in registration.get("participant_name", ""):
        return fail("registration.json still needs participant_name.")

    pr_text = pr_path.read_text(encoding="utf-8")
    if f"Registration UUID: {uuid}" not in pr_text:
        return fail("PR_DESCRIPTION.md must contain the same UUID as registration.json.")

    for path in [video_path, report_path, trajectory_path, policy_card_path, eval_path, narration_path, judge_brief_path, scorecard_path, manifest_path]:
        if not path.exists():
            return fail(f"Missing required artifact: {path.relative_to(ROOT)}")
        if path.stat().st_size <= 0:
            return fail(f"Artifact is empty: {path.relative_to(ROOT)}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("success"):
        return fail("Self-audit report success is false; regenerate or fix the demo.")
    if float(report.get("final_task_completion", 0.0)) < 1.0:
        return fail("Self-audit final_task_completion is below 1.0.")
    closed_loop = report.get("closed_loop_metrics", {})
    if closed_loop.get("controller") != "residual visual-servo/contact/slip policy":
        return fail("Report must include closed-loop residual controller metrics.")
    if float(closed_loop.get("median_visual_servo_error_m", 1.0)) > 0.012:
        return fail("Closed-loop median visual-servo error is too high.")
    if float(closed_loop.get("servo_error_reduction_pct", 0.0)) < 40.0:
        return fail("Residual controller must reduce visual-servo error by at least 40%.")
    if int(closed_loop.get("corrections_applied", 0)) <= 0:
        return fail("Closed-loop policy did not apply residual corrections.")

    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    required_fields = {
        "control_mode",
        "visual_servo_error_m",
        "raw_visual_servo_error_m",
        "contact_balance_error",
        "slip_observer_error_mm",
        "residual_action_norm",
        "policy_confidence",
        "feedback_correction_xyz",
    }
    missing = required_fields.difference(trajectory[0])
    if missing:
        return fail(f"Trajectory is missing closed-loop fields: {sorted(missing)}")

    policy_card = json.loads(policy_card_path.read_text(encoding="utf-8"))
    if "closed_loop_evidence" not in policy_card:
        return fail("Policy card must include closed_loop_evidence.")
    if int(policy_card.get("actuated_channels", 0)) < 22:
        return fail("Policy card must document at least 22 actuated channels.")
    if "five-finger" not in policy_card.get("hand_topology", ""):
        return fail("Policy card must document the five-finger hand topology.")

    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = evaluation.get("summary", {})
    if int(evaluation.get("rollout_count", 0)) < 32:
        return fail("Evaluation must include at least 32 fixed-seed stress rollouts.")
    if float(summary.get("residual_policy_success_rate", 0.0)) < 0.95:
        return fail("Residual policy stress-test success rate is below 95%.")
    if float(summary.get("median_improvement_mm", 0.0)) <= 10.0:
        return fail("Residual policy must beat the no-residual baseline by more than 10 mm median.")

    judge_brief = judge_brief_path.read_text(encoding="utf-8")
    for phrase in ["five-finger", "22-channel", "Residual-policy success", "Rubric Mapping"]:
        if phrase not in judge_brief:
            return fail(f"Judge brief is missing required phrase: {phrase}")

    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if len(scorecard.get("scorecard", {})) < 8:
        return fail("Rubric scorecard must cover all eight official scoring dimensions.")

    narration = narration_path.read_text(encoding="utf-8")
    if "Visual servoing aligns the palm" not in narration:
        return fail("Narration SRT must include the visual-servo story beat.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("registration_uuid") != uuid:
        return fail("submission_manifest.json UUID must match registration.json.")
    if manifest.get("canonical_pull_request") != "https://github.com/Faraday-Future-AI/Robothon-starter/pull/15":
        return fail("submission_manifest.json must point to canonical PR #15.")

    print("[ok] Dexterous Triage Lab submission package is internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
