#!/bin/bash
# Regenerate EVERY paper figure + CSV with the adopted rendering parameters.
# Run from the repo root after the experiments have written their JSONs:
#     bash plotting/make_figures.sh
# The parameters below are the paper's rendering decisions (axis windows,
# display cadences, panel geometry); the plot scripts'
# own defaults are deliberately generic.  Every figure re-derives from the raw
# run JSONs -- nothing here edits data.
set -e
PY=${PYTHON:-python}

# 1B main experiment (four muon arms): wide 4-column combined figure, y up
# to 3.7, x from step 1000.
# Display cadence: validation every 100 optimizer steps
# (every 4th recorded eval), training loss raw + 100-step moving average
# (TRAIN_SMOOTH in the scripts); the 1B ablation scripts below use the same
# defaults.
$PY plotting/plot_main_bench.py gpt-large --val-interval 100 \
    --ymax 3.7 --xmin-iter 1000 --figwidth 16 --aspect 0.55

# 10B ablation (four muon arms): continuous axis, 300-step display
# cadence for val; train loss uses the raw + 100-step moving-average rendering.
$PY plotting/plot_main_bench.py gpt-large \
    --root outputs/data10B_ablation --out figures/data10B_ablation \
    --ymax 3.25 --xmin-iter 5000 --val-interval 300 --figwidth 16 --aspect 0.55

# NS-iterations, cutoff: final-val summaries + per-method/per-arm curve sets
$PY plotting/plot_ns_iters.py gpt-large
$PY plotting/plot_cutoff_ablation.py gpt-large

# lr / momentum: per-tier combined figures (curve_layout.py,
# main-figure paradigm; x from step 1000 and figwidth 16 / aspect 0.55 are the
# scripts' defaults) + final-val summaries
$PY plotting/plot_lr_ablation.py gpt-large
$PY plotting/plot_momentum_ablation.py gpt-large

echo "all figures regenerated under figures/"
