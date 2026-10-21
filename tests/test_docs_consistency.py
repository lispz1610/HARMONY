#!/usr/bin/env python3
"""Documentation consistency checks for the Super Job variance investigation (review round 3).

Verifies that README/ADR vocabulary follows GLOSSARY.md, that documented artifact
names match what benchmarks/analyze_super_matrix.py and run_super_matrix.sh write,
that the README .gitignore inventory matches .gitignore, and that README
table-of-contents anchors resolve to real headings.
"""

import os
import re
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read(*parts):
    with open(os.path.join(ROOT_DIR, *parts), encoding="utf-8") as handle:
        return handle.read()


def github_anchor(heading):
    anchor = heading.strip().lower()
    anchor = re.sub(r"[^\w\- ]", "", anchor)
    return anchor.replace(" ", "-")


class TestGlossaryVocabulary(unittest.TestCase):
    def test_super_job_docs_avoid_glossary_avoid_terms(self):
        """Super Job docs and scripts must not use any term listed after '_Avoid_:' in GLOSSARY.md."""
        avoid_terms = []
        for line in read("GLOSSARY.md").splitlines():
            if line.startswith("_Avoid_:"):
                avoid_terms += [t.strip().lower() for t in line.split(":", 1)[1].split(",")]
        self.assertIn("orchestrator", avoid_terms)

        files = [
            ("docs", "adr", "0001-non-invasive-pbs-super-job.md"),
            ("job_harmony_super.pbs",),
            ("benchmarks", "run_super_matrix.sh"),
            ("benchmarks", "telemetry_probe.sh"),
            ("benchmarks", "analyze_super_matrix.py"),
        ]
        offenders = []
        for parts in files:
            for number, line in enumerate(read(*parts).splitlines(), 1):
                for term in avoid_terms:
                    if re.search(r"\b" + re.escape(term), line.lower()):
                        offenders.append(f"{'/'.join(parts)}:{number}: {term}")

        # README: the project title (an acronym expansion) predates this work and is exempt.
        readme_lines = read("README.md").splitlines()[1:]
        for number, line in enumerate(readme_lines, 2):
            if "orchestrat" in line.lower() and re.search(r"super_matrix|job_harmony_super|super job", line, re.I):
                offenders.append(f"README.md:{number}: orchestrat*")
        self.assertEqual(offenders, [])


class TestArtifactNames(unittest.TestCase):
    def test_readme_artifact_names_match_analyzer_outputs(self):
        """README must name the analyzer's real paired CSV and list hardware_telemetry.json."""
        analyzer = read("benchmarks", "analyze_super_matrix.py")
        readme = read("README.md")
        self.assertIn('output_dir / "super_matrix_paired.csv"', analyzer)
        self.assertIn('output_dir / "hardware_telemetry.json"', analyzer)
        self.assertNotIn("super_matrix_paired_ratios.csv", readme)
        self.assertIn("`super_matrix_paired.csv`", readme)
        section = readme.split("Artifacts generated in", 1)[1].split("Offline recovery", 1)[0]
        for name in (
            "SUPER_JOB_REPORT.md",
            "super_matrix_summary.csv",
            "super_matrix_paired.csv",
            "hardware_telemetry.log",
            "hardware_telemetry.json",
            "run_manifest.json",
        ):
            self.assertIn(name, section, f"{name} missing from README artifact list")

    def test_runner_writes_documented_raw_artifacts(self):
        runner = read("benchmarks", "run_super_matrix.sh")
        self.assertIn("hardware_telemetry.log", runner)
        self.assertIn("run_manifest.json", runner)


class TestGitignoreInventory(unittest.TestCase):
    def test_readme_gitignore_row_lists_python_cache_patterns(self):
        gitignore = read(".gitignore").splitlines()
        row = next(l for l in read("README.md").splitlines() if l.startswith("| `.gitignore`"))
        for pattern in ("__pycache__/", "*.py[cod]"):
            self.assertIn(pattern, gitignore)
            self.assertIn(pattern, row)


class TestReadmeAnchors(unittest.TestCase):
    def test_table_of_contents_anchors_resolve(self):
        readme = read("README.md")
        anchors = {github_anchor(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", readme, re.M)}
        broken = [
            a for a in re.findall(r"\]\(#([^)]+)\)", readme) if a not in anchors
        ]
        self.assertEqual(broken, [])


if __name__ == "__main__":
    unittest.main()
