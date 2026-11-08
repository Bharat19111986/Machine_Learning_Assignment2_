# Iris Classifier: End-to-End ML Deployment

An interactive Iris flower classifier demonstrating model training, held-out
evaluation, a prediction UI, and deployment with Streamlit.

## Run locally

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

`train.py` reports test accuracy and the confusion matrix. The app trains and
caches the same scikit-learn pipeline, displays its evaluation, and lets you
explore predictions by changing the four flower measurements in the sidebar.

## Deploy

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from the repository and select `app.py` as the entry point.
3. Deploy. Streamlit Cloud installs the packages from `requirements.txt`.

The model uses a stratified 80/20 train-test split with a fixed random seed,
feature standardization, and logistic regression. The Iris dataset is bundled
with scikit-learn, so no external dataset download is needed.