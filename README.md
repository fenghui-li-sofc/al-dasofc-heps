# GPR analysis of high-entropy perovskite anodes

Code and predictions for the study of ammonia conversion on high-entropy perovskite anodes. The package supports archived-result verification, holdout evaluation, and GPR training with an external training table.

# Data

The candidate space contains 4089 compositions: 76 training, 15 holdout, and 3998 remaining candidates. The computational A-site composition is La0.583Sr0.417.

## Run

Use Python 3.12 with the pinned dependencies, preferably in a virtual environment:
python -m pip install -r requirements.txt

These commands check archived files, recompute holdout metrics, and generate plots without retraining. Outputs are saved under `results/`.
Exact reproduction requires the original full-precision features, labels, and identifiers. The SI reports training compositions, labels, and rounds; these do not by themselves supply the complete feature matrix. Cross-validation output includes the supplied training labels.

# Method and provenance
Three independent GPR models use standardized features, normalized targets, and a Matérn kernel (nu = 2.5) plus white noise. Grouped outer folds evaluate generalization across B-site element combinations; grouped inner folds calibrate uncertainty. Numerical settings are retained in `model_core.py`.


## Data

The candidate space contains 4089 compositions: 76 training, 15 holdout, and 3998 remaining candidates. The computational A-site composition is La0.583Sr0.417.
