"""Run the Phase 3 baseline sweep for one seed as a preflight."""

from run_phase3_sweep import run_phase3_sweep


if __name__ == "__main__":
    run_phase3_sweep(seeds=[1001], max_epochs=5, min_epochs=3, patience=3)
