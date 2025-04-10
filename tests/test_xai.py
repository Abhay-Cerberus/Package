from AiPackageWrapper.xai_module import ExplainableAi
from unittest.mock import patch
import numpy as np

def mock_joblib_load(path):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    return model, X, X, y, y

@patch('joblib.load', side_effect=mock_joblib_load)
def test_explain_model_runs(mock_load):
    ExplainableAi.explain_model(None, None)