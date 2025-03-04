"""
Tests for the visualization module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from interview import visualization


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric1': [10, 20, 30, 40, 50],
        'numeric2': [5, 15, 25, 35, 45],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


def test_set_default_style():
    """Test set_default_style function."""
    # Save original rcParams
    original_figsize = plt.rcParams['figure.figsize']
    
    # Apply style
    visualization.set_default_style()
    
    # Check that rcParams were modified
    assert plt.rcParams['figure.figsize'] == (12, 8)
    assert plt.rcParams['font.size'] == 12
    
    # Restore original rcParams
    plt.rcParams['figure.figsize'] = original_figsize


def test_plot_distribution(sample_dataframe):
    """Test plot_distribution function."""
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Call the function
    result_ax = visualization.plot_distribution(
        sample_dataframe, 
        'numeric1',
        bins=5,
        title='Test Distribution',
        ax=ax
    )
    
    # Check that the function returns the axes
    assert result_ax is ax
    
    # Check that the title was set
    assert ax.get_title() == 'Test Distribution'
    
    # Close the figure to avoid memory leaks
    plt.close(fig)


def test_plot_categorical_distribution(sample_dataframe):
    """Test plot_categorical_distribution function."""
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Call the function
    result_ax = visualization.plot_categorical_distribution(
        sample_dataframe, 
        'category',
        title='Test Categorical Distribution',
        ax=ax
    )
    
    # Check that the function returns the axes
    assert result_ax is ax
    
    # Check that the title was set
    assert ax.get_title() == 'Test Categorical Distribution'
    
    # Close the figure to avoid memory leaks
    plt.close(fig)


def test_plot_scatter(sample_dataframe):
    """Test plot_scatter function."""
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Call the function
    result_ax = visualization.plot_scatter(
        sample_dataframe, 
        'numeric1',
        'numeric2',
        title='Test Scatter Plot',
        ax=ax
    )
    
    # Check that the function returns the axes
    assert result_ax is ax
    
    # Check that the title was set
    assert ax.get_title() == 'Test Scatter Plot'
    
    # Close the figure to avoid memory leaks
    plt.close(fig)


def test_plot_correlation_matrix(sample_dataframe):
    """Test plot_correlation_matrix function."""
    # Call the function
    ax = visualization.plot_correlation_matrix(
        sample_dataframe,
        columns=['numeric1', 'numeric2'],
        title='Test Correlation Matrix'
    )
    
    # Check that the function returns an axes
    assert isinstance(ax, plt.Axes)
    
    # Check that the title was set
    assert ax.get_title() == 'Test Correlation Matrix'
    
    # Close the figure to avoid memory leaks
    plt.close(plt.gcf())


def test_plot_boxplot(sample_dataframe):
    """Test plot_boxplot function."""
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Call the function
    result_ax = visualization.plot_boxplot(
        sample_dataframe, 
        'numeric1',
        'category',
        title='Test Box Plot',
        ax=ax
    )
    
    # Check that the function returns the axes
    assert result_ax is ax
    
    # Check that the title was set
    assert ax.get_title() == 'Test Box Plot'
    
    # Close the figure to avoid memory leaks
    plt.close(fig)


def test_create_subplot_grid():
    """Test create_subplot_grid function."""
    # Call the function
    fig, axes = visualization.create_subplot_grid(
        n_plots=3,
        n_cols=2,
        figsize=(10, 8)
    )
    
    # Check that the function returns a figure and axes
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    
    # Check that the correct number of subplots were created
    assert len(axes.flatten()) == 4  # 2x2 grid for 3 plots
    
    # Check that the unused subplot is hidden
    assert not axes.flatten()[-1].get_visible()
    
    # Close the figure to avoid memory leaks
    plt.close(fig)