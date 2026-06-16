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

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    uuid = registration.get("uuid", "")
    if not UUID_RE.match(uuid):
        return fail("registration.json still needs a valid Robothon UUID.")
    if "PASTE-" in registration.get("participant_name", ""):
        return fail("registration.json still needs participant_name.")

    pr_text = pr_path.read_text(encoding="utf-8")
    if f"Registration UUID: {uuid}" not in pr_text:
        return fail("PR_DESCRIPTION.md must contain the same UUID as registration.json.")

    for path in [video_path, report_path, trajectory_path]:
        if not path.exists():
            return fail(f"Missing required artifact: {path.relative_to(ROOT)}")
        if path.stat().st_size <= 0:
            return fail(f"Artifact is empty: {path.relative_to(ROOT)}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("success"):
        return fail("Self-audit report success is false; regenerate or fix the demo.")
    if float(report.get("final_task_completion", 0.0)) < 1.0:
        return fail("Self-audit final_task_completion is below 1.0.")

    print("[ok] Dexterous Triage Lab submission package is internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
