import unittest
import numpy as np
from har_edu_analytics import SensorFeatureExtractor, HARClassifier, EducationalRetentionModel

class TestHAREdu(unittest.TestCase):
    def test_pipeline(self):
        feats = SensorFeatureExtractor.extract_window_features(np.random.randn(32, 3), np.random.randn(32, 3))
        self.assertEqual(len(feats), 26)
        har = HARClassifier().benchmark_har_classification()
        self.assertGreater(har.accuracy, 0.90)
        ret = EducationalRetentionModel().evaluate_retention_cohort()
        self.assertGreater(ret.model_auc, 0.90)

if __name__ == "__main__":
    unittest.main()
