"""
NeuroTraffic ML Pipeline — Master Runner
==========================================
Runs the ML pipeline in the correct order.

Usage (Docker cluster):
  spark-submit --master local[*] --driver-memory 4g run_pipeline.py --hdfs

Usage (local testing):
  spark-submit run_pipeline.py --local /path/to/combined_v3

"""

import subprocess
import sys
import time
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1. Contamination Tuning",    "tune_contamination.py"),
    ("2. Main Pipeline (IF + GBT)", "disruption_detector.py"),
    ("3. Scoring Module Test",      "scoring_module.py"),
]

# Pass through flags (--hdfs or --local /path)
extra_args = sys.argv[1:]

print("=" * 60)
print("NeuroTraffic ML Pipeline")
print(f"Args: {' '.join(extra_args) if extra_args else '(none)'}")
print("=" * 60)

start_step = 1
if "--step" in sys.argv:
    idx = sys.argv.index("--step")
    start_step = int(sys.argv[idx + 1])
    extra_args = [a for a in extra_args if a not in ("--step", sys.argv[idx + 1])]

for i, (name, script) in enumerate(STEPS, 1):
    if i < start_step:
        print(f"  SKIP  {name}")
        continue

    script_path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {name}")
    print(f"{'=' * 60}")

    t0 = time.time()
    cmd = ["spark-submit", script_path] + extra_args
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        print(f"  Re-run with: spark-submit run_pipeline.py --step {i} {' '.join(extra_args)}")
        sys.exit(1)

    print(f"  DONE: {name} ({elapsed:.0f}s)")

print(f"\n{'=' * 60}")
print("ALL STEPS COMPLETE")
print("=" * 60)
print("\nFor the streaming pipeline:")
print("  from scoring_module import NeuroTrafficScorer")
print("  scorer = NeuroTrafficScorer(spark)")
print("  result = scorer.full_pipeline(incoming_batch)")
