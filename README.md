# Two-Feature Class-Centroid Decision Trees: Code, Benchmarks, and Experimental Results

This repository provides the reproducible code, benchmark manifest, synthetic task generators, aggregate experimental tables, and figures for the manuscript *"When do two-feature centroid splits help? Geometry, compactness and failure modes of restricted oblique trees"*.

## Repository Contents

- `src/models.py`: Implementation of the two-feature class-centroid tree (`P2`) and diagnostic variants (`P2-H`, `P2-Tuned`, `P-Dense`, `P2-Random`, `P2-Unscaled`, `LDA-Tree`, `Logistic-Tree`, and `Random-Projection`).
- `src/prepare_data.py`: OpenML dataset download/preparation, exact-duplicate-group partitioning, and preprocessing pipeline definitions.
- `src/run_experiments.py`: Primary 35-task benchmark evaluation (20 methods, 5 repeated group-disjoint splits) and 6-task ablation suite.
- `src/run_supplement.py`: Controlled-geometry synthetic task generator and one-factor sensitivity experiments (`diagonal`, `axis`, `imbalanced`, `circle`, `xor`, `dense`).
- `src/run_matched.py`: Fixed-depth capacity diagnostic (`depth_limit` in `{1, 2, 3, 4, 6}`) and fresh-seed synthetic geometry replication (seeds `10--14`).
- `src/analyze.py`: Metric recomputation, dataset-level bootstrap confidence intervals, Holm-adjusted paired Wilcoxon tests, omnibus Friedman test, numerical perturbation evaluation, and path-preservation radius calculations.
- `src/plot_results.py` and `src/build_tables.py`: Figure and LaTeX table generation directly from saved CSV/JSON results.
- `src/audit_artifacts.py`: Integrity verification across predictions, split disjointness, and generated artifacts.
- `src/lht_adapter.cpp` and `src/run_lht.py`: Adapter and diagnostic harness used for the separate Learning Hyperplane Tree (LHT) implementation audit.
- `exp/`: Saved aggregate CSV/JSON tables, dataset manifest (`exp/data/manifest.csv` with OpenML IDs, URLs, and SHA-256 hashes), and environment freeze records (`exp/metadata/environment_freeze.txt`).
- `figures/`: Generated manuscript figures in PDF and PNG formats.

## Datasets

All 35 public classification tasks are available from OpenML (`https://www.openml.org/d/<id>`). Dataset IDs, canonical OpenML URLs, observation counts, predictor compositions, and SHA-256 checksums are recorded in `exp/data/manifest.csv` and `exp/data/<id>.json`.

## Reproduction

Experiments were executed on Python 3.11.10 (CPU only, 4 worker processes for primary benchmarks, 1 numerical-library thread per process). Dependencies are listed in `requirements-research.txt` and pinned in `exp/metadata/environment_freeze.txt`.

To reproduce the benchmark evaluation, analysis, tables, and figures from the workspace root:

```bash
python src/prepare_data.py
python src/run_experiments.py --workers 4
python src/run_experiments.py --suite ablation --datasets 11,37,50,1462,1510,40983 --methods P2-Tuned,P-Dense,P2-Random,P2-Unscaled --workers 2
python src/run_supplement.py --workers 2
python src/run_matched.py
python src/analyze.py
python src/plot_results.py
python src/build_tables.py
python src/audit_artifacts.py
```

Note on execution provenance: per-run raw model binaries (`exp/runs/`), cached raw dataset pickles (`exp/data/*.pkl`, `exp/data/*.npz`), and third-party C++ sources are excluded from this repository to respect storage limits and third-party redistribution terms; all aggregate tables, statistical summaries, and reproduction scripts are retained. An initial driver invocation metadata overwrite occurred prior to adding per-invocation metadata preservation and is recorded in the archived environment logs.

## License

Code in this repository is released under the MIT License (`LICENSE`). Public OpenML datasets remain subject to their respective original source licenses.
