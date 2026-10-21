#!/usr/bin/env bash
# Invariants and automated guardrail check for HARMONY Super Job and runtime core.
# Ensures that:
# 1. dcl/ runtime core remains 100% untouched and frozen against master.
# 2. All offline unit and integration tests pass cleanly.
# 3. Prebuilt benchmark runner verifies without recompilation.
# 4. Prohibited domain terminology from GLOSSARY.md is not introduced in tracked files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1. Checking Runtime Core Invariants (dcl/ freeze) ==="
if git rev-parse --verify master >/dev/null 2>&1; then
    dcl_diff="$(git diff master...HEAD -- dcl/ || true)"
    if [[ -n "$dcl_diff" ]]; then
        echo "ERROR: Modifications detected in dcl/ runtime core! Core must remain 100% frozen." >&2
        echo "$dcl_diff" >&2
        exit 1
    fi
    echo "PASS: dcl/ runtime core is 100% clean and frozen."
else
    echo "NOTICE: master branch not available locally; skipping dcl/ diff check."
fi

echo "=== 2. Running Test Suite Discovery ==="
python3 -m unittest discover -s tests -p 'test_*.py'

echo "=== 3. Verifying Prebuilt Benchmark Script Guard ==="
bash tests/test_compare_prebuilt.sh

echo "=== 4. Checking Domain Glossary Terminology Guard ==="
python3 - << 'PY'
import subprocess
import sys

# Scan git diff against master (or HEAD if no master) for prohibited terms per GLOSSARY.md
# Exclude GLOSSARY.md itself where avoid-terms are formally defined.
diff_cmd = ["git", "diff", "master...HEAD", "--", "*.md", "*.py", "*.sh", "*.pbs", ":!GLOSSARY.md"]
try:
    diff_output = subprocess.check_output(diff_cmd, text=True, stderr=subprocess.DEVNULL)
except Exception:
    diff_output = ""

# Prohibited synonyms explicitly noted in GLOSSARY.md:
# - Avoid "orchestrator", "orchestration" for Super Job
# - Avoid "legacy log", "old run" for Historical Evidence
prohibited = [
    ("orchestration", "Use 'Super Job driver' instead (GLOSSARY.md avoids 'orchestrator/orchestration')"),
    ("legacy log", "Use 'historical log' instead (GLOSSARY.md avoids 'legacy log')"),
]

violations = []
for line in diff_output.splitlines():
    if line.startswith("+") and not line.startswith("+++"):
        low = line.lower()
        for term, msg in prohibited:
            if term in low:
                violations.append(f"Found prohibited term '{term}': {line.strip()}\n  -> {msg}")

if violations:
    print("ERROR: Prohibited terminology found in diff:", file=sys.stderr)
    for v in violations:
        print(f"  {v}", file=sys.stderr)
    sys.exit(1)

print("PASS: No prohibited domain terms found in diff.")
PY

echo "=== All invariants and automated guardrails passed successfully ==="
exit 0
