"""
Tests for the models module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

from interview import models


@pytest.fixture
def classification_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        random_state=42
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data():
    """Create sample regression data for testing."""
    X, y = make_regression(
        n_samples=100,
        n_features=4,
        n_informative=2,
        random_state=42
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric1': [10, 20, 30, 40, 50],
        'numeric2': [5, 15, 25, 35, 45],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


def test_preprocess_features(sample_dataframe):
    """Test preprocess_features function."""
    # Split the DataFrame
    train_df = sample_dataframe.iloc[:3]
    test_df = sample_dataframe.iloc[3:]
    
    # Call the function
    X_train_processed, X_test_processed, preprocessor = models.preprocess_features(
        train_df,
        test_df,
        categorical_features=['category'],
        numerical_features=['numeric1', 'numeric2']
    )
    
    # Check that the function returns numpy arrays and a preprocessor
    assert isinstance(X_train_processed, np.ndarray)
    assert isinstance(X_test_processed, np.ndarray)
    assert hasattr(preprocessor, 'transform')
    
    # Check that the arrays have the correct shape
    # 3 samples, 2 numeric features + 3 one-hot encoded categories (A, B, C)
    assert X_train_processed.shape[0] == 3
    assert X_train_processed.shape[1] == 5
    
    # 2 samples, same number of features
    assert X_test_processed.shape[0] == 2
    assert X_test_processed.shape[1] == 5


def test_encode_target(classification_data):
    """Test encode_target function."""
    _, _, y_train, y_test = classification_data
    
    # Call the function
    y_train_encoded, y_test_encoded, encoder = models.encode_target(y_train, y_test)
    
    # Check that the function returns numpy arrays and an encoder
    assert isinstance(y_train_encoded, np.ndarray)
    assert isinstance(y_test_encoded, np.ndarray)
    assert isinstance(encoder, LabelEncoder)
    
    # Check that the arrays have the correct shape
    assert y_train_encoded.shape == y_train.shape
    assert y_test_encoded.shape == y_test.shape


def test_get_classification_models():
    """Test get_classification_models function."""
    # Call the function
    model_dict = models.get_classification_models()
    
    # Check that the function returns a dictionary
    assert isinstance(model_dict, dict)
    
    # Check that the dictionary contains the expected models
    assert 'Logistic Regression' in model_dict
    assert 'Decision Tree' in model_dict
    assert 'Random Forest' in model_dict
    assert 'SVM' in model_dict
    assert 'KNN' in model_dict
    
    # Check that the models are valid scikit-learn estimators
    for model in model_dict.values():
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


def test_get_regression_models():
    """Test get_regression_models function."""
    # Call the function
    model_dict = models.get_regression_models()
    
    # Check that the function returns a dictionary
    assert isinstance(model_dict, dict)
    
    # Check that the dictionary contains the expected models
    assert 'Linear Regression' in model_dict
    assert 'Ridge' in model_dict
    assert 'Lasso' in model_dict
    assert 'Decision Tree' in model_dict
    assert 'Random Forest' in model_dict
    assert 'Gradient Boosting' in model_dict
    assert 'SVR' in model_dict
    
    # Check that the models are valid scikit-learn estimators
    for model in model_dict.values():
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


def test_train_and_evaluate_classifier(classification_data):
    """Test train_and_evaluate_classifier function."""
    X_train, X_test, y_train, y_test = classification_data
    
    # Call the function with default model
    model, metrics = models.train_and_evaluate_classifier(
        X_train, X_test, y_train, y_test
    )
    
    # Check that the function returns a model and metrics
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert isinstance(metrics, dict)
    
    # Check that the metrics dictionary contains the expected metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Call the function with custom model
    custom_model = LogisticRegression(C=0.1, random_state=42)
    model, metrics = models.train_and_evaluate_classifier(
        X_train, X_test, y_train, y_test, custom_model=custom_model
    )
    
    # Check that the returned model is the custom model
    assert model is custom_model


def test_train_and_evaluate_regressor(regression_data):
    """Test train_and_evaluate_regressor function."""
    X_train, X_test, y_train, y_test = regression_data
    
    # Call the function with default model
    model, metrics = models.train_and_evaluate_regressor(
        X_train, X_test, y_train, y_test
    )
    
    # Check that the function returns a model and metrics
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert isinstance(metrics, dict)
    
    # Check that the metrics dictionary contains the expected metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    
    # Call the function with custom model
    custom_model = LinearRegression()
    model, metrics = models.train_and_evaluate_regressor(
        X_train, X_test, y_train, y_test, custom_model=custom_model
    )
    
    # Check that the returned model is the custom model
    assert model is custom_model


def test_plot_confusion_matrix(classification_data):
    """Test plot_confusion_matrix function."""
    _, _, _, y_test = classification_data
    
    # Create some predictions
    y_pred = np.random.choice(np.unique(y_test), size=len(y_test))
    
    # Call the function
    ax = models.plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=['Class 0', 'Class 1', 'Class 2']
    )
    
    # Check that the function returns an axes
    assert isinstance(ax, plt.Axes)
    
    # Close the figure to avoid memory leaks
    plt.close(plt.gcf())


def test_compare_models(classification_data, regression_data):
    """Test compare_models function."""
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = classification_data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = regression_data
    
    # Get classification models
    classification_models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Dummy Classifier': LogisticRegression(C=0.001, random_state=42)
    }
    
    # Call the function for classification
    results_cls = models.compare_models(
        X_train_cls, X_test_cls, y_train_cls, y_test_cls,
        classification_models, is_classification=True
    )
    
    # Check that the function returns a DataFrame
    assert isinstance(results_cls, pd.DataFrame)
    
    # Check that the DataFrame has the expected columns
    assert 'Model' in results_cls.columns
    assert 'CV Score' in results_cls.columns
    assert 'Test Accuracy' in results_cls.columns
    
    # Get regression models
    regression_models = {
        'Linear Regression': LinearRegression(),
        'Dummy Regressor': LinearRegression()
    }
    
    # Call the function for regression
    results_reg = models.compare_models(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        regression_models, is_classification=False
    )
    
    # Check that the function returns a DataFrame
    assert isinstance(results_reg, pd.DataFrame)
    
    # Check that the DataFrame has the expected columns
    assert 'Model' in results_reg.columns
    assert 'CV Score (neg MSE)' in results_reg.columns
    assert 'Test MSE' in results_reg.columns


def test_plot_model_comparison():
    """Test plot_model_comparison function."""
    # Create a sample results DataFrame
    results = pd.DataFrame({
        'Model': ['Model A', 'Model B', 'Model C'],
        'Accuracy': [0.8, 0.7, 0.9],
        'MSE': [0.2, 0.3, 0.1]
    })
    
    # Call the function for classification
    ax_cls = models.plot_model_comparison(
        results, 'Accuracy', is_classification=True
    )
    
    # Check that the function returns an axes
    assert isinstance(ax_cls, plt.Axes)
    
    # Close the figure to avoid memory leaks
    plt.close(plt.gcf())
    
    # Call the function for regression
    ax_reg = models.plot_model_comparison(
        results, 'MSE', is_classification=False
    )
    
    # Check that the function returns an axes
    assert isinstance(ax_reg, plt.Axes)
    
    # Close the figure to avoid memory leaks
    plt.close(plt.gcf())