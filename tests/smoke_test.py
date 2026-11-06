"""End-to-end smoke test on stand-in calls. Writes to tests/_out, never to reports/.

    python -m tests.smoke_test
"""
import shutil
import sys
from pathlib import Path

import run_all
from src import data, llm_extract
from tests.make_fixtures import write

ROOT = Path(__file__).resolve().parent
FIX, OUT = ROOT / "fixtures", ROOT / "_out"

if __name__ == "__main__":
    shutil.rmtree(FIX, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    write(FIX)
    data.DATA_DIR = FIX
    run_all.REPORT_DIR = OUT
    llm_extract.CACHE_DIR = OUT / "cache"
    sys.argv = ["run_all.py", "--llm", "mock"]
    run_all.main()
    expected = ["summary.md", "outcome_overall.csv", "outcome_accuracy_by_field.png", "phase_confusion_real.png"]
    missing = [f for f in expected if not (OUT / f).exists()]
    assert not missing, f"missing outputs: {missing}"
    print("\nSMOKE TEST PASSED. Outputs in", OUT)
