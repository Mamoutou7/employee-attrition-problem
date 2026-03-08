# Employee Attrition Problem

A small Python machine learning project for predicting employee attrition (`left`) from HR data.

## Overview

This repository trains a binary classifier on an employee dataset (`sourcedata/hr_data_comma_sep.csv`) to estimate whether an employee is likely to leave the company.

Current training code in `mlib.py` supports:
- Logistic Regression
- Linear SVM
- Gradient Boosting
- Random Forest

The project also includes basic Google App Engine deployment files:
- `app.yaml`
- `cloudbuild.yaml`

## Project Structure

```text
.
├── sourcedata/              # Raw HR dataset
├── eda/                     # Exploratory analysis folder
├── logs/                    # Training logs
├── mlib.py                  # Training logic
├── main.py                  # Entry point placeholder
├── requirements.txt         # Python dependencies
├── Makefile                 # Install/lint helpers
├── app.yaml                 # App Engine config
└── cloudbuild.yaml          # Cloud Build deploy config
```

## Dataset

The dataset contains HR features such as:
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `promotion_last_5years`
- `department`
- `salary`
- target: `left`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with Make:

```bash
make install
```

## Run Training

The training script is implemented in `mlib.py`:

```bash
python mlib.py
```

By default, it trains a Random Forest model and saves it as `trainedmodel.pkl`.

## Notes

This repository is a partial training/deployment template rather than a fully runnable production app.
Some files referenced by the code, such as `config.json`, are not included, and `main.py` is currently empty.

## License

This project includes a `LICENSE` file in the repository.
