# Reproduction notes

The reported manuscript numbers are stored as CSV/JSON artifacts under `analysis/`. The following files map directly to the revised claims:

- `analysis/full_scale/transfer_matrix_exact_accuracy.csv`: 4x4 transfer matrix.
- `analysis/full_scale/evaluation_summary.csv`: exact accuracy and reliability summaries.
- `analysis/full_scale/vscore_table.csv` and `vscore_sensitivity.csv`: V-Score and all-row sensitivity calculations.
- `analysis/full_scale/ood_entropy_summary.csv`: entropy OOD results, including below-chance AUROC values.
- `analysis/full_scale/temperature_scaling_summary.csv`: calibration comparison.
- `analysis/seed_pilot/`: five-seed reduced fixed-slot summaries.
- `analysis/honesty/`: verifier outputs and the corrected-renderer audit.
- `crnn_pilot/crnn_transfer_pilot.csv`: reduced CRNN convergence/transfer check.

The manuscript source can be compiled from `manuscript/` with Tectonic or an equivalent LaTeX toolchain. The response source can be compiled from `response/`. The checked-in PDFs are the exact files supplied in the revision package.
