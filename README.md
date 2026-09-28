# CAPTCHA-X revision artifact

This repository contains the anonymized review artifact for the revised manuscript **Benchmark Collapse in Text CAPTCHAs: Generator Heterogeneity, Cross-Generator Transfer, and Reliability-Aware Synthetic Evaluation**.

The artifact is intended to make the reported analyses inspectable during peer review. It contains the revised manuscript and response PDFs, LaTeX sources, figures, aggregate analysis tables, generator-honesty reports, the reduced five-seed pilot, verifier and reliability scripts, and the source-level benchmark components used by the reported experiments.

## Scope and evidence limits

All images in the study are locally synthesized. No live CAPTCHA service, production CAPTCHA data, or human participant data is included. The full-scale matrix uses the fixed-slot classifier and seed 17. The five-seed results are a reduced fixed-slot pilot. The reduced CRNN attempt is retained as a negative convergence check; it does not establish architecture-invariant transfer or collapse claims. The original pre-repair image archive and verifier log were not preserved, so the available captcha-library row is labeled as a proxy.

## Layout

- `manuscript/`: revised manuscript source, figures, and PDF.
- `response/`: anonymized point-by-point response source and PDF.
- `analysis/`: full-scale tables, seed-pilot summaries, and honesty-audit reports.
- `generator_code/`, `models/`, `evaluation/`, `experiments/`: source-level benchmark components.
- `scripts/`: verifier, reliability, V-Score, failure-taxonomy, and CRNN-pilot helpers.
- `configs/`: journal full-scale and reduced seed-sweep configurations.

## Reproduction

The aggregate artifacts can be inspected without regenerating images. To run the basic smoke checks in a Python environment with the dependencies in `requirements.txt`:

```bash
python test_basic.py
```

The configurations use relative paths and do not include machine-specific directories. Full image generation and model training require the declared Python dependencies and substantial compute/storage; no claim is made that the reduced CRNN pilot converged.

## Anonymity

The manuscript and response use anonymous author metadata. The repository history uses a neutral artifact commit identity. The hosting account is visible in the repository URL because GitHub does not provide anonymous ownership under the submitting account; no author names, affiliations, emails, or local paths are included in the artifact contents.
