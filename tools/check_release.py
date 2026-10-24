"""Check the release allowlist and obvious private/training material."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"README.md", "LICENSE", "pyproject.toml", ".gitignore",
           ".github/workflows/tests.yml", "docs/INPUTS_AND_METHODS.md",
           "docs/RELEASE_SCOPE.md", "examples/make_synthetic.py",
           "examples/comparisons.json", "examples/slides_template.csv",
           "tools/check_release.py", "tests/test_evaluation.py", "tests/test_inference.py",
           "src/mscl_eval/__init__.py", "src/mscl_eval/io.py", "src/mscl_eval/metrics.py",
           "src/mscl_eval/evaluation.py", "src/mscl_eval/plotting.py", "src/mscl_eval/cli.py",
           "src/mscl_eval/inference.py", "src/mscl_eval/spatial.py"}
PATTERNS = {
    "private absolute path": r"/(?:home|Users)/[A-Za-z0-9_.-]+/",
    "hospital identifier": r"\b(?:B\d{7,10}|Y\d{7,10})\b",
    "TCGA participant identifier": r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\b",
    "GitHub credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "private key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    "gradient update": r"\.backward\s*\(",
    "optimiser implementation": r"torch\.optim\.",
}


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode().split("\0")
    names = {name for name in names if name}
    errors = [f"File not on release allowlist: {name}" for name in sorted(names - ALLOWED)]
    errors += [f"Missing required file: {name}" for name in sorted(ALLOWED - names)]
    for name in sorted(names & ALLOWED):
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"Symlink is not allowed: {name}")
            continue
        if path.stat().st_size > 200_000:
            errors.append(f"Unexpectedly large source file: {name}")
        text = path.read_text(encoding="utf-8")
        for label, pattern in PATTERNS.items():
            if re.search(pattern, text):
                errors.append(f"{label}: {name}")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"PASS: {len(names)} allowlisted text files; no matching private-data/credential/training patterns.")
    print("This automated check supplements, but does not replace, a manual release review.")


if __name__ == "__main__":
    main()
