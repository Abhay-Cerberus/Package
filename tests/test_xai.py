import pytest
import joblib
from AiPackageWrapper.xai_module import ExplainableAi  # Adjust import if module name is different

def test_explain_model_runs(monkeypatch):
    # Load data from the model file
    model, X_train, X_test, y_train, y_test = joblib.load('churn_model.pkl')

    # Patch SHAP plotting to avoid opening actual plots during test
    monkeypatch.setattr("shap.plots.waterfall", lambda x: None)
    monkeypatch.setattr("shap.plots.beeswarm", lambda x: None)

    # Call the explain_model method and ensure no exceptions are raised
    try:
        ExplainableAi.explain_model(model, X_test)
    except Exception as e:
        pytest.fail(f"explain_model raised an exception: {e}")