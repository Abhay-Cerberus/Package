import numpy as np
import pytest
from unittest.mock import patch

# Define a dummy callable model (so that shap.Explainer doesn't complain)
def dummy_model(X):
    # Return the input unchanged; this function is now callable.
    return X

# This mock will replace joblib.load and return a tuple as expected.
def mock_joblib_load(_):
    X_train = np.random.rand(10, 3)
    X_test = np.random.rand(5, 3)
    y_train = np.random.randint(0, 2, size=10)
    y_test = np.random.randint(0, 2, size=5)
    # Return a callable model along with dummy train and test data.
    return dummy_model, X_train, X_test, y_train, y_test

# A dummy explainer that returns a dummy shap value for any index access.
class DummyExplainer:
    def __init__(self, model):
        self.model = model

    def __call__(self, data):
        class DummySHAPValues:
            def __init__(self):
                self.base_values = [0.5]
                self.data = data
                self.values = [[0.1, -0.2, 0.3]]

            def __getitem__(self, idx):
                return self

        return DummySHAPValues()

# Patch joblib.load so it returns our dummy model/data.
# Patch shap.Explainer so it returns a DummyExplainer instance.
# Patch the shap plotting functions to no-ops.
@patch("joblib.load", side_effect=mock_joblib_load)
@patch("shap.Explainer", return_value=DummyExplainer(dummy_model))
@patch("shap.plots.waterfall", lambda x: None)
@patch("shap.plots.beeswarm", lambda x: None)
def test_explain_model_runs(mock_explainer, mock_load):
    from AiPackageWrapper.xai_module import ExplainableAi
    try:
        # The real code calls joblib.load('churn_model.pkl') inside explain_model,
        # so our patch will intercept that call.
        ExplainableAi.explain_model(None, None)
    except Exception as e:
        pytest.fail(f"ExplainableAi.explain_model raised an exception: {e}")
