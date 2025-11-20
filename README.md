
ICE-NODE PyTorch Reproduction
=============================

This repository contains a PyTorch reproduction of the
"Integration of Clinical Embeddings with Neural Ordinary Differential Equations"
(ICE-NODE) model from MLHC 2022, adapted to run on MIMIC-III using
PyTorch and `torchdiffeq`.


Getting Started
---------------

### 1. Create a Python environment

You can use any environment manager; for example with `uv`:

```bash
cd ice-node-repro
uv venv .venv
source .venv/bin/activate
uv sync
```

Make sure you also install `torchdiffeq` and `scikit-learn` if they are not
already present in your environment.


### 2. Prepare data

You need access to the MIMIC-III data (following the PhysioNet license).

1. Configure paths to your local MIMIC-III CSVs in `etl.py`.
2. Run the preprocessing script to build patient time series and splits:

```bash
uv run etl.py --dataset mimic3
```

This will create processed pickles under `data/processed/`.

3. Build final training examples (per split) from the time series:

```bash
uv run build_examples.py --dataset mimic3
```


### 3. Train models

The main training script is `train3.py`. Example commands:

**ICE-NODE (ICENodeAugmented) with focal loss**

```bash
uv run train3.py \
  --model ICENodeAugmented \
  --dataset mimic3 \
  --epochs 60 \
  --batch-size 256 \
  --lr-dynamics 7.15e-5 \
  --lr-other 1.14e-3 \
  --reg-alpha 50.0 \
  --use-focal-loss
```

**GRU baseline**

```bash
uv run train3.py \
  --model GRUBaseline \
  --dataset mimic3 \
  --epochs 30 \
  --batch-size 256 \
  --lr-other 1e-3 \
  --use-focal-loss \
  --no-regularization
```

Checkpoints are written to `checkpoints/` as
`{dataset}_{model}_best.pt`, selected by best validation micro AUC.


### 4. Evaluate on test set

Use `eval.py` to compute the same metrics as in
`ICE-NODE/notebooks/dx_fine_analysis.ipynb`:

```bash
uv run eval.py \
  --model-path checkpoints/mimic3_ICENodeAugmented_best.pt \
  --dataset mimic3
```

This script reports micro/macro AUROC and AUPRC, hits@K, and the
ACC-Pi-kK detectability metrics stratified by code frequency
quantiles, and writes results to `eval_results/`.


Citation
--------

If you use this code or ideas from it in your work, please cite the
original ICE-NODE paper and codebase:

- ICE-NODE paper:

  > Julio A. Castillo-Barahona, Arthur Jacot, Barbara Engelhardt,
  > Mauricio A. Álvarez, "Integration of Clinical Embeddings with
  > Neural Ordinary Differential Equations", MLHC 2022.

- Original JAX implementation:

  > ICE-NODE: Integration of Clinical Embeddings with Neural Ordinary
  > Differential Equations, barahona-research-group/ICE-NODE,
  > https://github.com/barahona-research-group/ICE-NODE

