#!/usr/bin/env python3
"""
Unit and integration tests for the PBS Super Job script (job_harmony_super.pbs).
Verifies:
  - Bash syntax validity (bash -n)
  - Required PBS directives (name, output, walltime, vnodes chunks, place)
  - LIMC module loading rules (no module purge, correct module loads)
  - Prebuilt binary verification (fail fast on compute node)
  - Nodefile slot count verification (exactly 2 slots)
  - Preflight health checks (topo_probe and test_async_halo_opencl)
  - Output directory isolation and pipeline execution
"""

import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOB_SCRIPT_PATH = os.path.join(ROOT_DIR, "job_harmony_super.pbs")
MOCK_LAUNCHER_PATH = os.path.join(ROOT_DIR, "tests", "mock_cluster_launcher.py")
KNEIGHBOR_CPP_PATH = os.path.join(ROOT_DIR, "benchmarks", "kneighbor.cpp")


class TestSuperJobScriptDirectivesAndSyntax(unittest.TestCase):
    """Verifies static properties of job_harmony_super.pbs."""

    def test_file_exists(self):
        """job_harmony_super.pbs must exist at repository root."""
        self.assertTrue(
            os.path.isfile(JOB_SCRIPT_PATH),
            f"Expected {JOB_SCRIPT_PATH} to exist at repo root",
        )

    def test_bash_syntax(self):
        """job_harmony_super.pbs must have valid bash syntax."""
        proc = subprocess.run(
            ["bash", "-n", JOB_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"bash -n failed on {JOB_SCRIPT_PATH}:\n{proc.stderr}",
        )

    def test_pbs_directives(self):
        """Verifies required PBS directives in job_harmony_super.pbs."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.startswith("#PBS")]

        directives = " ".join(lines)
        self.assertIn("#PBS -N harmony_super", directives)
        self.assertIn("#PBS -j oe", directives)
        self.assertIn("#PBS -o harmony_super.out", directives)
        self.assertIn("#PBS -l walltime=02:00:00", directives)
        self.assertIn("#PBS -l place=free", directives)

        # Check vnode chunk specification
        expected_select = (
            "#PBS -l select=1:ncpus=4:mpiprocs=1:mem=8gb:ngpus=1:vnode=compute-1-0[0]"
            "+1:ncpus=4:mpiprocs=1:mem=8gb:ngpus=1:vnode=compute-1-1[0]"
        )
        self.assertIn(expected_select, directives)

    def test_module_loading_rules(self):
        """Verifies LIMC module loading rules: no module purge, required modules loaded."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Must NOT execute module purge
        self.assertFalse(
            re.search(r"^\s*module\s+purge", content, re.MULTILINE),
            "job_harmony_super.pbs must not execute 'module purge' as it breaks OpenHPC / LIMC defaults",
        )

        # Must load required modules
        self.assertTrue(
            re.search(r"module\s+load\s+gnu13/13\.2\.0", content),
            "Expected 'module load gnu13/13.2.0'",
        )
        self.assertTrue(
            re.search(r"module\s+load\s+openmpi5/5\.0\.5", content),
            "Expected 'module load openmpi5/5.0.5'",
        )
        self.assertTrue(
            re.search(r"module\s+load\s+prun/2\.2", content),
            "Expected 'module load prun/2.2'",
        )

    def test_analyzer_invocation_not_masked(self):
        """job_harmony_super.pbs must not mask analyze_super_matrix.py with || true."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn(
            "|| true",
            re.findall(r".*analyze_super_matrix\.py.*", content)[0] if "analyze_super_matrix.py" in content else "",
            "job_harmony_super.pbs must not mask analyzer failure with '|| true'",
        )
        self.assertIn("analyzer_exit_code", content)

    def test_readme_documents_super_job_workflow(self):
        """README.md must document the complete 5-step Super Job workflow."""
        readme_path = os.path.join(ROOT_DIR, "README.md")
        self.assertTrue(os.path.isfile(readme_path), "README.md must exist")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("job_harmony_super.pbs", readme)
        self.assertIn("cluster_build.sh", readme)
        self.assertIn("qsub job_harmony_super.pbs", readme)
        self.assertIn("qstat -u $USER", readme)
        self.assertIn("results/variance_run_", readme)
        self.assertIn("SUPER_JOB_REPORT.md", readme)
        self.assertIn("analyze_super_matrix.py", readme)
        self.assertIn("--summarize-only", readme)


class TestSuperJobExecution(unittest.TestCase):
    """Dynamic execution tests for job_harmony_super.pbs with mock environments."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_super_job_")
        self.mock_bin_dir = os.path.join(self.temp_dir, "bin")
        self.cluster_build_dir = os.path.join(self.temp_dir, "build", "cluster")
        os.makedirs(self.mock_bin_dir, exist_ok=True)
        os.makedirs(self.cluster_build_dir, exist_ok=True)

        # Create mock nodefile with 2 slots
        self.nodefile = os.path.join(self.temp_dir, "pbs_nodefile")
        with open(self.nodefile, "w", encoding="utf-8") as f:
            f.write("compute-1-0\ncompute-1-1\n")

        # Create mock module command
        self.mock_module = os.path.join(self.mock_bin_dir, "module")
        with open(self.mock_module, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(self.mock_module, 0o755)

        # Create mock prun command pointing to mock_cluster_launcher.py
        self.mock_prun = os.path.join(self.mock_bin_dir, "prun")
        with open(self.mock_prun, "w", encoding="utf-8") as f:
            f.write(f'#!/bin/sh\nexec python3 "{MOCK_LAUNCHER_PATH}" "$@"\n')
        os.chmod(self.mock_prun, 0o755)

        # Setup default environment
        self.env = os.environ.copy()
        self.env["PATH"] = f"{self.mock_bin_dir}:{self.env.get('PATH', '')}"
        self.env["PBS_O_WORKDIR"] = self.temp_dir
        self.env["PBS_NODEFILE"] = self.nodefile
        self.env["PBS_JOBID"] = "12345.headnode"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_cluster_binaries(self, binaries=None):
        """Creates dummy executable cluster binaries in build/cluster."""
        if binaries is None:
            binaries = [
                "topo_probe.out",
                "test_async_halo_opencl.out",
                "kneighbor.out",
                "leanmd.out",
            ]
        for b in binaries:
            path = os.path.join(self.cluster_build_dir, b)
            with open(path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)

    def _run_job_script(self, custom_env=None, env_override=False):
        """Executes job_harmony_super.pbs with bash."""
        if env_override and custom_env is not None:
            run_env = custom_env.copy()
        else:
            run_env = self.env.copy()
            if custom_env:
                run_env.update(custom_env)
        return subprocess.run(
            ["bash", JOB_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.temp_dir,
            env=run_env,
        )

    def test_missing_pbs_env_vars_fails(self):
        """Job script must fail if PBS_O_WORKDIR or PBS_NODEFILE is missing."""
        env_missing_workdir = self.env.copy()
        del env_missing_workdir["PBS_O_WORKDIR"]
        proc = self._run_job_script(env_missing_workdir, env_override=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PBS_O_WORKDIR", proc.stderr + proc.stdout)

        env_missing_nodefile = self.env.copy()
        del env_missing_nodefile["PBS_NODEFILE"]
        proc = self._run_job_script(env_missing_nodefile, env_override=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PBS_NODEFILE", proc.stderr + proc.stdout)

    def test_invalid_nodefile_slot_count_fails(self):
        """Job script must fail if PBS_NODEFILE does not contain exactly 2 slots."""
        with open(self.nodefile, "w", encoding="utf-8") as f:
            f.write("compute-1-0\n")  # Only 1 slot

        self._create_cluster_binaries()
        proc = self._run_job_script()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("two", (proc.stderr + proc.stdout).lower())

    def test_missing_prebuilt_binaries_fails_fast(self):
        """Job script must fail fast if prebuilt cluster binaries are missing, citing cluster_build.sh."""
        # Empty build/cluster directory
        proc = self._run_job_script()
        self.assertNotEqual(proc.returncode, 0)
        combined_output = proc.stderr + proc.stdout
        self.assertIn("Missing", combined_output)
        self.assertIn("cluster_build.sh", combined_output)

    def test_missing_single_binary_fails(self):
        """Job script must fail if any one required binary is missing."""
        self._create_cluster_binaries(["topo_probe.out", "test_async_halo_opencl.out", "kneighbor.out"])
        # leanmd.out is missing
        proc = self._run_job_script()
        self.assertNotEqual(proc.returncode, 0)
        combined_output = proc.stderr + proc.stdout
        self.assertIn("leanmd.out", combined_output)
        self.assertIn("cluster_build.sh", combined_output)

    def test_preflight_topo_probe_failure_aborts(self):
        """Job script must abort if preflight topo_probe fails or reports synthetic data."""
        self._create_cluster_binaries()
        proc = self._run_job_script({"MOCK_FAIL_TOPO_PROBE": "1"})
        self.assertNotEqual(proc.returncode, 0)

    def test_preflight_halo_test_failure_aborts(self):
        """Job script must abort if preflight test_async_halo_opencl fails."""
        self._create_cluster_binaries()
        proc = self._run_job_script({"MOCK_FAIL_PREFLIGHT_HALO": "1"})
        self.assertNotEqual(proc.returncode, 0)

    def test_asymmetric_gpu_allocation_fails_preflight(self):
        """Job script must abort if preflight encounters asymmetric GPU allocation."""
        self._create_cluster_binaries()
        proc = self._run_job_script({"MOCK_ASYMMETRIC_GPUS": "1"})
        self.assertNotEqual(proc.returncode, 0)
        output = proc.stderr + proc.stdout
        self.assertIn("ERROR: Asynchronous halo test preflight failed (requires at least 1 OpenCL GPU per rank)", output)

    def test_runner_invocation_passes_skip_preflight(self):
        """job_harmony_super.pbs must pass --skip-preflight to run_super_matrix.sh."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("--skip-preflight", content, "job_harmony_super.pbs should pass --skip-preflight to run_super_matrix.sh")

    def test_successful_mock_pipeline_execution(self):
        """Job script runs through preflight, runner, and analyzer producing expected outputs."""
        self._create_cluster_binaries()

        # Also copy benchmarks/ scripts into temp_dir so script finds them relative to PBS_O_WORKDIR
        shutil.copytree(
            os.path.join(ROOT_DIR, "benchmarks"),
            os.path.join(self.temp_dir, "benchmarks"),
        )

        custom_env = {
            "BENCH_SEEDS": "42",
            "BENCH_ITERATIONS": "5",
        }
        proc = self._run_job_script(custom_env)
        self.assertEqual(
            proc.returncode,
            0,
            f"Super Job execution failed with error:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

        expected_output_dir = os.path.join(self.temp_dir, "results", "variance_run_12345.headnode")
        self.assertTrue(
            os.path.isdir(expected_output_dir),
            f"Expected output directory {expected_output_dir} to exist",
        )

        # Verify output artifacts exist
        report_md = os.path.join(expected_output_dir, "SUPER_JOB_REPORT.md")
        self.assertTrue(
            os.path.isfile(report_md),
            f"Expected {report_md} to be generated by post-processor",
        )
        topo_json = os.path.join(expected_output_dir, "topo_metrics_cluster.json")
        self.assertTrue(
            os.path.isfile(topo_json),
            f"Expected {topo_json} to exist",
        )

    def test_sample_failure_propagates_exit_code_and_runs_postprocessing(self):
        """When a matrix sample fails, PBS job exits 1 but runs post-processing to generate report."""
        self._create_cluster_binaries()

        shutil.copytree(
            os.path.join(ROOT_DIR, "benchmarks"),
            os.path.join(self.temp_dir, "benchmarks"),
        )

        fail_marker = os.path.join(self.temp_dir, "fail_once.marker")
        custom_env = {
            "BENCH_SEEDS": "42",
            "BENCH_ITERATIONS": "5",
            "MOCK_FAIL_ONCE_FILE": fail_marker,
        }
        proc = self._run_job_script(custom_env)
        self.assertEqual(proc.returncode, 1, "PBS job script must exit 1 when a sample fails")

        expected_output_dir = os.path.join(self.temp_dir, "results", "variance_run_12345.headnode")
        self.assertTrue(os.path.isdir(expected_output_dir))

        report_md = os.path.join(expected_output_dir, "SUPER_JOB_REPORT.md")
        self.assertTrue(
            os.path.isfile(report_md),
            f"Expected {report_md} to be generated by post-processor even on sample failure",
        )

    def test_analyzer_failure_propagates_exit_code(self):
        """When post-processing analyzer fails, PBS job must exit with code 1."""
        self._create_cluster_binaries()

        shutil.copytree(
            os.path.join(ROOT_DIR, "benchmarks"),
            os.path.join(self.temp_dir, "benchmarks"),
        )

        # Overwrite analyze_super_matrix.py to simulate failure
        analyzer_script = os.path.join(self.temp_dir, "benchmarks", "analyze_super_matrix.py")
        with open(analyzer_script, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('Mock analyzer failed\\n')\nsys.exit(1)\n")
        os.chmod(analyzer_script, 0o755)

        custom_env = {
            "BENCH_SEEDS": "42",
            "BENCH_ITERATIONS": "5",
        }
        proc = self._run_job_script(custom_env)
        self.assertEqual(
            proc.returncode,
            1,
            f"PBS job script must exit 1 when analyzer fails, got {proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        self.assertIn("analyzer failed", (proc.stderr + proc.stdout).lower())


class TestKNeighborNumericalVerification(unittest.TestCase):
    """Verifies that benchmarks/kneighbor.cpp rejects non-finite outputs (NaN and Inf)."""

    def test_kneighbor_source_checks_isfinite(self):
        """kneighbor.cpp must check std::isfinite on actual and reference values."""
        self.assertTrue(
            os.path.isfile(KNEIGHBOR_CPP_PATH),
            f"Expected {KNEIGHBOR_CPP_PATH} to exist",
        )
        with open(KNEIGHBOR_CPP_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        expected_check = "!std::isfinite(out_data[i]) || !std::isfinite(cpu_val) || std::abs(out_data[i] - cpu_val) > 1e-5f"
        self.assertIn(
            expected_check,
            content,
            f"kneighbor.cpp must check isfinite on out_data[i] and cpu_val. Expected to find:\n{expected_check}",
        )
        self.assertNotIn(
            "if (std::abs(out_data[i] - cpu_val) > 1e-5f) {",
            content,
            "kneighbor.cpp must not use naive std::abs check without isfinite guards",
        )

    def test_verification_rejects_nan_and_inf(self):
        """Numerical verification logic must reject NaN, Inf, and large deltas, while accepting finite matches."""
        def check_mismatch(actual: float, cpu_val: float) -> bool:
            # Replicates the C++ logic: returns True if mismatch (verification failed)
            return (
                not math.isfinite(actual)
                or not math.isfinite(cpu_val)
                or abs(actual - cpu_val) > 1e-5
            )

        # Finite matches should not trigger mismatch
        self.assertFalse(check_mismatch(1.0, 1.0))
        self.assertFalse(check_mismatch(0.0, 0.0))
        self.assertFalse(check_mismatch(1.000001, 1.0))

        # Finite mismatches should trigger mismatch
        self.assertTrue(check_mismatch(1.0, 1.01))
        self.assertTrue(check_mismatch(1.0, 0.0))

        # NaN values must trigger mismatch
        self.assertTrue(check_mismatch(float("nan"), 1.0))
        self.assertTrue(check_mismatch(1.0, float("nan")))
        self.assertTrue(check_mismatch(float("nan"), float("nan")))

        # Inf values must trigger mismatch
        self.assertTrue(check_mismatch(float("inf"), 1.0))
        self.assertTrue(check_mismatch(1.0, float("inf")))
        self.assertTrue(check_mismatch(float("inf"), float("inf")))
        self.assertTrue(check_mismatch(float("-inf"), float("-inf")))
        self.assertTrue(check_mismatch(float("-inf"), 1.0))

    def test_cpp_verification_compilation_and_rejection(self):
        """Compile and execute a test harness exercising kneighbor.cpp's verification check with C++ IEEE-754 semantics."""
        with open(KNEIGHBOR_CPP_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the exact condition used in kneighbor.cpp inside the verification loop
        match = re.search(r"float cpu_val = [^;]+;\s*\n\s*if \((.+?)\) \{", content)
        self.assertIsNotNone(match, "Could not find stencil verification check in kneighbor.cpp")
        cond_str = match.group(1).strip()

        # Build a small C++ program testing this exact condition on non-finite and finite inputs
        cpp_src = f"""#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

bool verify_sample(float actual, float cpu_val) {{
    float out_data[1] = {{ actual }};
    std::size_t i = 0;
    if ({cond_str}) {{
        return false; // rejected / mismatch detected
    }}
    return true; // accepted
}}

int main() {{
    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float inf_val = std::numeric_limits<float>::infinity();

    // Valid matches must pass
    if (!verify_sample(1.0f, 1.0f)) return 10;
    if (!verify_sample(0.5f, 0.5f)) return 11;

    // Numerical differences must fail
    if (verify_sample(1.0f, 2.0f)) return 20;

    // Non-finite values MUST fail (return false from verify_sample)
    if (verify_sample(nan_val, 1.0f)) return 30;
    if (verify_sample(1.0f, nan_val)) return 31;
    if (verify_sample(nan_val, nan_val)) return 32;
    if (verify_sample(inf_val, 1.0f)) return 33;
    if (verify_sample(1.0f, inf_val)) return 34;
    if (verify_sample(inf_val, inf_val)) return 35;
    if (verify_sample(-inf_val, 1.0f)) return 36;
    if (verify_sample(-inf_val, -inf_val)) return 37;

    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "test_check.cpp")
            bin_file = os.path.join(tmpdir, "test_check.out")
            with open(src_file, "w", encoding="utf-8") as f:
                f.write(cpp_src)

            compile_proc = subprocess.run(
                ["g++", "-std=c++20", "-O2", src_file, "-o", bin_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                compile_proc.returncode,
                0,
                f"Compilation failed for C++ verification test:\n{compile_proc.stderr}",
            )

            run_proc = subprocess.run([bin_file], capture_output=True, text=True)
            self.assertEqual(
                run_proc.returncode,
                0,
                f"C++ verification check failed to reject non-finite values (exit code {run_proc.returncode})",
            )


if __name__ == "__main__":
    unittest.main()
