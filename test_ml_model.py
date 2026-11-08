import unittest

from ml_model import train_and_evaluate


class ModelWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.evaluation = train_and_evaluate()

    def test_evaluation_uses_all_classes_and_test_samples(self):
        self.assertGreaterEqual(self.evaluation["accuracy"], 0.9)
        self.assertEqual(self.evaluation["confusion_matrix"].shape, (3, 3))
        self.assertEqual(self.evaluation["confusion_matrix"].sum(), 30)

    def test_model_predicts_probabilities_for_each_species(self):
        prediction = self.model.predict([[5.1, 3.5, 1.4, 0.2]])
        probabilities = self.model.predict_proba([[5.1, 3.5, 1.4, 0.2]])

        self.assertEqual(prediction.shape, (1,))
        self.assertEqual(prediction[0], 0)
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(probabilities.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
