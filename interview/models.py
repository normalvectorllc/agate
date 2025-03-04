"""
Model building and evaluation utilities for the Pokemon dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: List[str] = None,
    numerical_features: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features using scikit-learn's ColumnTransformer.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        categorical_features (List[str], optional): List of categorical feature names. 
                                                   Defaults to None.
        numerical_features (List[str], optional): List of numerical feature names. 
                                                 Defaults to None.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed training and testing features
    """
    if categorical_features is None:
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    if numerical_features is None:
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform the testing data
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, preprocessor


def encode_target(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode categorical target variable.
    
    Args:
        y_train (np.ndarray): Training target values
        y_test (np.ndarray): Testing target values
    
    Returns:
        Tuple[np.ndarray, np.ndarray, LabelEncoder]: Encoded training and testing targets,
                                                    and the encoder
    """
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    return y_train_encoded, y_test_encoded, encoder


def get_classification_models() -> Dict[str, Any]:
    """
    Get a dictionary of classification models.
    
    Returns:
        Dict[str, Any]: Dictionary mapping model names to model instances
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }


def get_regression_models() -> Dict[str, Any]:
    """
    Get a dictionary of regression models.
    
    Returns:
        Dict[str, Any]: Dictionary mapping model names to model instances
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR()
    }


def train_and_evaluate_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'Random Forest',
    custom_model: Any = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train and evaluate a classification model.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training target values
        y_test (np.ndarray): Testing target values
        model_name (str, optional): Name of the model to use. Defaults to 'Random Forest'.
        custom_model (Any, optional): Custom model instance. Defaults to None.
    
    Returns:
        Tuple[Any, Dict[str, float]]: Trained model and evaluation metrics
    """
    if custom_model is not None:
        model = custom_model
    else:
        models = get_classification_models()
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
        model = models[model_name]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return model, metrics


def train_and_evaluate_regressor(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'Random Forest',
    custom_model: Any = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train and evaluate a regression model.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training target values
        y_test (np.ndarray): Testing target values
        model_name (str, optional): Name of the model to use. Defaults to 'Random Forest'.
        custom_model (Any, optional): Custom model instance. Defaults to None.
    
    Returns:
        Tuple[Any, Dict[str, float]]: Trained model and evaluation metrics
    """
    if custom_model is not None:
        model = custom_model
    else:
        models = get_regression_models()
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
        model = models[model_name]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    normalize: bool = False
) -> plt.Axes:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        class_names (Optional[List[str]], optional): List of class names. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
        cmap (str, optional): Colormap. Defaults to 'Blues'.
        normalize (bool, optional): Whether to normalize values. Defaults to False.
    
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    
    # Plot the heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd', 
        cmap=cmap,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Labels and title
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    return ax


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Axes:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model (Any): Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
        top_n (int, optional): Number of top features to show. Defaults to 10.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
    
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Limit to top_n features
    indices = indices[:top_n]
    
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top Feature Importances')
    
    return plt.gca()


def compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    is_classification: bool = True,
    cv: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training target values
        y_test (np.ndarray): Testing target values
        models (Dict[str, Any]): Dictionary of models to compare
        is_classification (bool, optional): Whether this is a classification task. 
                                           Defaults to True.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
    
    Returns:
        pd.DataFrame: DataFrame with model comparison results
    """
    results = []
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if is_classification:
            cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            test_score = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'Model': name,
                'CV Score': cv_score.mean(),
                'Test Accuracy': test_score,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        else:
            cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'CV Score (neg MSE)': cv_score.mean(),
                'Test MSE': mse,
                'Test RMSE': rmse,
                'Test MAE': mae,
                'R² Score': r2
            })
    
    return pd.DataFrame(results)


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str,
    is_classification: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Axes:
    """
    Plot model comparison results.
    
    Args:
        results (pd.DataFrame): DataFrame with model comparison results
        metric (str): Metric to plot
        is_classification (bool, optional): Whether this is a classification task. 
                                           Defaults to True.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
    
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    plt.figure(figsize=figsize)
    
    # Sort by the specified metric
    sorted_results = results.sort_values(by=metric, ascending=not is_classification)
    
    # Create bar plot
    ax = sns.barplot(x='Model', y=metric, data=sorted_results)
    
    # Add value labels on top of bars
    for i, v in enumerate(sorted_results[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return ax