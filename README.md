# Rainfall Prediction with WeatherAUS

## Problem Statement

This project predicts whether it will rain tomorrow (`RainTomorrow`: `Yes` or `No`) using Australian weather observations. Five classification models are trained and compared using a held-out test set. The Streamlit app evaluates saved models against labeled CSV data; it does not retrain models on uploaded data.

## Dataset Description

The public [WeatherAUS dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) contains 142,193 daily observations from Australian weather stations, with 22 input features such as location, temperature, rainfall, wind, and humidity. The target, `RainTomorrow`, records whether it rained the following day. This project trains on a stratified sample of 20,000 rows and evaluates on a separate sample of 5,000.

## GitHub Repository

[Machine_Learning_Assignment2_](https://github.com/Bharat19111986/Machine_Learning_Assignment2_)

## Models and Evaluation

All metrics below are calculated on the same 5,000-row holdout sample. `Yes` is treated as the positive class for AUC, precision, recall, and F1.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8062 | 0.8821 | 0.5466 | 0.7948 | 0.6478 | 0.5377 |
| Decision Tree | 0.7828 | 0.7937 | 0.5113 | 0.7074 | 0.5936 | 0.4616 |
| K-Nearest Neighbors | 0.8416 | 0.8567 | 0.7519 | 0.4380 | 0.5536 | 0.4904 |
| Gaussian Naive Bayes | 0.6606 | 0.7549 | 0.3691 | 0.7244 | 0.4890 | 0.3080 |
| Support Vector Machine | 0.8608 | 0.8820 | 0.7728 | 0.5370 | 0.6337 | 0.5651 |

### Performance Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Best recall (0.7948) and marginally best AUC (0.8821), so it identifies the largest share of rainy days but has more false-positive predictions. |
| Decision Tree | Lower accuracy and AUC than the other models, with moderate recall; it is the weakest tree-based result in this comparison. |
| K-Nearest Neighbors | Strong precision (0.7519) and accuracy (0.8416), but the lowest recall among the non-Naive Bayes models (0.4380), missing many rainy days. |
| Gaussian Naive Bayes | Highest recall after Logistic Regression (0.7244), but lowest accuracy, precision, F1, AUC, and MCC. |
| Support Vector Machine | Best accuracy (0.8608) and MCC (0.5651), with high precision and an AUC close to Logistic Regression; it is the strongest overall by accuracy and MCC. |

**Overall winner:** Support Vector Machine by accuracy and MCC. Logistic Regression is preferable when detecting rainy days (recall) is the priority.

## Run Locally

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

To retrain the models, place `weatherAUS.csv` at `data/weatherAUS.csv` and run:

```bash
python train_models.py
```

Start the interactive app:

```bash
streamlit run app.py
```

The app starts with the included `test_data.csv`. An uploaded labeled CSV must contain all WeatherAUS input columns and the `RainTomorrow` target. Select a model to view its metrics, confusion matrix, and classification report; the app also displays a comparison of all five saved models.

## Project Files

- `app.py`: Streamlit evaluation and comparison interface.
- `train_models.py`: stratified data split, preprocessing, training, and evaluation.
- `weather_ml.py`: shared feature engineering.
- `models/`: saved classifier pipelines, metadata, and training metrics.
- `test_data.csv`: labeled holdout data used by the app.
- `data/weatherAUS.csv`: source dataset required to retrain the models.