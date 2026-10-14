"""Run offline dependency/API checks for all learning stages. No training."""
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "learning").glob("*/exercise.py"))
    scripts += sorted((root / "learning").glob("*/*/exercise.py"))
    assert len(scripts) == 11, f"Expected 11 stage scripts, got {len(scripts)}"
    for path in scripts:
        print(path.relative_to(root), flush=True)
        module = runpy.run_path(str(path))
        module["check_environment"]()
    print("PASS: 11 stage environment checks. TODO training implementations were not checked.")


if __name__ == "__main__":
    main()
