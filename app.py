import streamlit as st

from ml_model import train_and_evaluate


@st.cache_resource
def get_model():
    return train_and_evaluate()


st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")
model, evaluation = get_model()
class_names = evaluation["class_names"]

st.title("🌸 Iris flower classifier")
st.write(
    "Explore a complete machine-learning workflow: train and evaluate a model, "
    "then use flower measurements to make a prediction."
)

with st.sidebar:
    st.header("Flower measurements")
    sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, 5.8, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.4, 3.0, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 6.9, 3.8, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
probabilities = model.predict_proba(
    [[sepal_length, sepal_width, petal_length, petal_width]]
)[0]

prediction_column, evaluation_column = st.columns(2)
with prediction_column:
    st.subheader("Prediction")
    st.success(f"Predicted species: **{class_names[prediction]}**")
    st.caption("Model confidence")
    st.progress(float(probabilities[prediction]))
    st.write(f"{probabilities[prediction]:.1%}")
    st.caption("Probabilities by species")
    for name, probability in zip(class_names, probabilities):
        st.write(f"{name}: {probability:.1%}")

with evaluation_column:
    st.subheader("Held-out evaluation")
    st.metric("Test accuracy", f"{evaluation['accuracy']:.1%}")
    st.caption("Confusion matrix (rows: actual, columns: predicted)")
    st.dataframe(
        {
            "Actual species": class_names,
            **{
                f"Predicted {name}": evaluation["confusion_matrix"][:, index]
                for index, name in enumerate(class_names)
            },
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.caption(
    "The logistic-regression pipeline is trained on the Iris dataset with a "
    "stratified 80/20 train-test split. Move the sliders in the sidebar to explore predictions."
)
