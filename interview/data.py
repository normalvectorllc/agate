"""
Data loading and preprocessing utilities for the Pokemon dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def get_dataset_path() -> Path:
    """
    Get the path to the Pokemon dataset.
    
    Returns:
        Path: Path to the Pokemon dataset directory
    """
    # Get the path relative to the package
    base_dir = Path(__file__).parent.parent
    return base_dir / "datasets" / "pokemon"


def load_pokemon_data() -> pd.DataFrame:
    """
    Load the Pokemon dataset into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: The Pokemon dataset
    """
    dataset_path = get_dataset_path() / "pokemon.csv"
    return pd.read_csv(dataset_path)


def clean_pokemon_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Pokemon dataset by handling missing values and converting data types.
    
    Args:
        df (pd.DataFrame): The raw Pokemon dataset
        
    Returns:
        pd.DataFrame: The cleaned Pokemon dataset
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical missing values with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df


def get_pokemon_types(df: pd.DataFrame) -> List[str]:
    """
    Get a list of unique Pokemon types from the dataset.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        
    Returns:
        List[str]: List of unique Pokemon types
    """
    if 'type1' in df.columns:
        return sorted(df['type1'].unique().tolist())
    return []


def get_pokemon_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate basic statistics for numeric columns in the Pokemon dataset.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary with column names as keys and
                                     statistics as values
    """
    stats = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max(),
            'std': df[col].std()
        }
    
    return stats


def filter_pokemon_by_type(df: pd.DataFrame, pokemon_type: str) -> pd.DataFrame:
    """
    Filter the Pokemon dataset by type.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        pokemon_type (str): The Pokemon type to filter by
        
    Returns:
        pd.DataFrame: Filtered Pokemon dataset
    """
    if 'type1' in df.columns:
        return df[df['type1'] == pokemon_type]
    return df


def get_feature_matrix(
    df: pd.DataFrame, 
    features: List[str], 
    target: Optional[str] = None
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Extract feature matrix and target vector from the Pokemon dataset.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        features (List[str]): List of feature column names
        target (Optional[str]): Target column name, if None only features are returned
        
    Returns:
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]: Feature matrix and target vector
                                                          if target is provided, otherwise
                                                          just the feature matrix
    """
    X = df[features].values
    
    if target is not None and target in df.columns:
        y = df[target].values
        return X, y
    
    return X