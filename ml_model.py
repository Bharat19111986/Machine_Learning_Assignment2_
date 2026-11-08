from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate():
    dataset = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)

    evaluation = {
        "accuracy": accuracy_score(labels_test, predictions),
        "confusion_matrix": confusion_matrix(
            labels_test, predictions, labels=range(len(dataset.target_names))
        ),
        "class_names": dataset.target_names,
        "feature_names": dataset.feature_names,
    }
    return model, evaluation
