from ml_model import train_and_evaluate


def main():
    _, evaluation = train_and_evaluate()
    print(f"Test accuracy: {evaluation['accuracy']:.1%}")
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(evaluation["confusion_matrix"])


if __name__ == "__main__":
    main()
