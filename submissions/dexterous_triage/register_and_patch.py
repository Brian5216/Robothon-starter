from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTER_URL = "https://robothon.ff.com/api/register"
UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def register(nickname: str, email: str, agent: str, direction: str) -> dict:
    payload = {
        "nickname": nickname,
        "email": email,
        "agent": agent,
        "direction": direction,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        REGISTER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"registration failed: HTTP {exc.code}: {detail}") from exc


def patch_files(uuid: str, participant_name: str) -> None:
    registration_path = ROOT / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["uuid"] = uuid
    registration["participant_name"] = participant_name
    registration["project_name"] = "Dexterous Triage Lab"
    registration_path.write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")

    pr_path = ROOT / "PR_DESCRIPTION.md"
    pr_text = pr_path.read_text(encoding="utf-8")
    pr_text = re.sub(r"Registration UUID: .*", f"Registration UUID: {uuid}", pr_text, count=1)
    pr_text = pr_text.replace("- [ ] `submissions/dexterous_triage/registration.json` contains my UUID", "- [x] `submissions/dexterous_triage/registration.json` contains my UUID")
    pr_text = pr_text.replace("- [ ] PR description contains the same UUID", "- [x] PR description contains the same UUID")
    pr_path.write_text(pr_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register for FFAI Robothon, then patch this submission with the returned UUID."
    )
    parser.add_argument("--nickname", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--participant-name", required=True)
    parser.add_argument("--agent", default="Codex")
    parser.add_argument(
        "--direction",
        default="dexterous hand / long-horizon tasks / data collection",
    )
    args = parser.parse_args()

    response = register(args.nickname, args.email, args.agent, args.direction)
    uuid = str(response.get("uuid", ""))
    if not UUID_RE.fullmatch(uuid):
        raise SystemExit(f"registration response did not include a valid UUID: {response}")

    patch_files(uuid, args.participant_name)
    validate = subprocess.run(
        [sys.executable, str(ROOT / "validate_submission.py")],
        cwd=str(ROOT.parents[1]),
        text=True,
        check=False,
    )
    print(json.dumps({"uuid": uuid, "duplicate": bool(response.get("duplicate"))}, indent=2))
    return validate.returncode


if __name__ == "__main__":
    sys.exit(main())
