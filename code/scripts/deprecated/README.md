# Deprecated Scripts

This directory stores historical one-off or superseded Ctrl-World training
helpers. They are kept for provenance and reproducibility, but new experiments
should not extend these scripts directly.

Current active entry points stay in `code/scripts/`:

- `launch_training.sh`
- `train_delta_ee.py`
- `precompute_latents_delta_ee.py`
- `precompute_latents_s1_pca.py`
- `compute_stat_family_roots.py`
- `backfill_ee_targets.py`
- validation scripts

Historical recipes moved here:

- S1-A / S1-B / S1-C training wrappers
- single-task S1 wrapper scripts
- all50 chunk16 wrapper
- old checkpoint watcher scripts
- older S1/enhanced stat and precompute helpers
