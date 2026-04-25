# Active Learning Framework for High-Entropy Perovskite Screening

This repository provides the active learning workflow used for screening high-entropy perovskite electrocatalysts.

## Overview

The framework combines:

- Gaussian Process Regression (GPR)
- Bayesian optimization for hyperparameter tuning
- Uncertainty-guided active learning

It enables efficient prediction of target descriptors with a limited number of DFT-labeled samples.

## Target Properties

The same workflow is applied to three descriptors:

- Oxygen vacancy formation energy (ΔE_Ov)
- Nitrogen adsorption energy (ΔE_N)
- Hydrogen adsorption energy (ΔE_H)

Only the target label and minor hyperparameters differ.  
The core algorithm and workflow remain identical.

## Usage

1. Prepare input datasets:
   - `Train_data.xlsx`: initial labeled data
   - `Sample.xlsx`: unlabeled candidate pool

2. Set target label in `main.py`:

```python
"target_label": "E_Ov"   # or E_N / E_H
