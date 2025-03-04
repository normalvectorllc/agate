"""
Tests for the data module.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from interview import data


def test_get_dataset_path():
    """Test get_dataset_path function."""
    path = data.get_dataset_path()
    assert isinstance(path, Path)
    assert path.name == "pokemon"
    assert path.parent.name == "datasets"


@pytest.mark.skipif(not os.path.exists(data.get_dataset_path() / "pokemon.csv"),
                   reason="Pokemon dataset not found")
def test_load_pokemon_data():
    """Test load_pokemon_data function."""
    df = data.load_pokemon_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_clean_pokemon_data():
    """Test clean_pokemon_data function."""
    # Create a test DataFrame with missing values
    test_df = pd.DataFrame({
        'A': [1, 2, None, 4, 5],
        'B': [None, 'b', 'c', 'd', None],
        'C': [1.1, 2.2, 3.3, None, 5.5]
    })
    
    cleaned_df = data.clean_pokemon_data(test_df)
    
    # Check that the original DataFrame is not modified
    assert pd.isna(test_df['A'].iloc[2])
    
    # Check that numeric columns have no missing values
    assert not cleaned_df['A'].isna().any()
    assert not cleaned_df['C'].isna().any()
    
    # Check that categorical columns have no missing values
    assert not cleaned_df['B'].isna().any()
    
    # Check that missing categorical values are filled with 'Unknown'
    assert cleaned_df['B'].iloc[0] == 'Unknown'
    assert cleaned_df['B'].iloc[4] == 'Unknown'


def test_get_pokemon_types():
    """Test get_pokemon_types function."""
    # Create a test DataFrame with type1 column
    test_df = pd.DataFrame({
        'type1': ['Fire', 'Water', 'Grass', 'Fire', 'Electric']
    })
    
    types = data.get_pokemon_types(test_df)
    
    assert isinstance(types, list)
    assert len(types) == 4  # Unique types
    assert 'Fire' in types
    assert 'Water' in types
    assert 'Grass' in types
    assert 'Electric' in types
    
    # Test with DataFrame without type1 column
    test_df2 = pd.DataFrame({
        'Type': ['Fire', 'Water', 'Grass']
    })
    
    types2 = data.get_pokemon_types(test_df2)
    assert isinstance(types2, list)
    assert len(types2) == 0


def test_get_pokemon_stats():
    """Test get_pokemon_stats function."""
    # Create a test DataFrame with numeric columns
    test_df = pd.DataFrame({
        'attack': [50, 60, 70, 80, 90],
        'defense': [40, 50, 60, 70, 80],
        'type': ['Fire', 'Water', 'Grass', 'Fire', 'Electric']
    })
    
    stats = data.get_pokemon_stats(test_df)
    
    assert isinstance(stats, dict)
    assert 'attack' in stats
    assert 'defense' in stats
    assert 'type' not in stats  # Not a numeric column
    
    # Check stats for attack
    assert 'mean' in stats['attack']
    assert 'median' in stats['attack']
    assert 'min' in stats['attack']
    assert 'max' in stats['attack']
    assert 'std' in stats['attack']
    
    assert stats['attack']['mean'] == 70.0
    assert stats['attack']['median'] == 70.0
    assert stats['attack']['min'] == 50.0
    assert stats['attack']['max'] == 90.0


def test_filter_pokemon_by_type():
    """Test filter_pokemon_by_type function."""
    # Create a test DataFrame with type1 column
    test_df = pd.DataFrame({
        'type1': ['Fire', 'Water', 'Grass', 'Fire', 'Electric'],
        'attack': [50, 60, 70, 80, 90]
    })
    
    filtered_df = data.filter_pokemon_by_type(test_df, 'Fire')
    
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) == 2
    assert filtered_df['type1'].iloc[0] == 'Fire'
    assert filtered_df['type1'].iloc[1] == 'Fire'
    
    # Test with non-existent type
    filtered_df2 = data.filter_pokemon_by_type(test_df, 'Dragon')
    assert isinstance(filtered_df2, pd.DataFrame)
    assert len(filtered_df2) == 0
    
    # Test with DataFrame without type1 column
    test_df2 = pd.DataFrame({
        'Type': ['Fire', 'Water', 'Grass']
    })
    
    filtered_df3 = data.filter_pokemon_by_type(test_df2, 'Fire')
    assert isinstance(filtered_df3, pd.DataFrame)
    assert len(filtered_df3) == 3  # Returns the original DataFrame


def test_get_feature_matrix():
    """Test get_feature_matrix function."""
    # Create a test DataFrame
    test_df = pd.DataFrame({
        'attack': [50, 60, 70, 80, 90],
        'defense': [40, 50, 60, 70, 80],
        'type': ['Fire', 'Water', 'Grass', 'Fire', 'Electric']
    })
    
    # Test with features only
    X = data.get_feature_matrix(test_df, ['attack', 'defense'])
    
    assert isinstance(X, np.ndarray)
    assert X.shape == (5, 2)
    
    # Test with features and target
    X, y = data.get_feature_matrix(test_df, ['attack'], 'defense')
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (5, 1)
    assert y.shape == (5,)
    
    # Test with non-existent target
    X_only = data.get_feature_matrix(test_df, ['attack'], 'NonExistent')
    
    assert isinstance(X_only, np.ndarray)
    assert X_only.shape == (5, 1)