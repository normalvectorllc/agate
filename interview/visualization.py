"""
Visualization utilities for the Pokemon dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


def set_default_style():
    """
    Set default style for matplotlib plots.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_distribution(
    df: pd.DataFrame, 
    column: str, 
    bins: int = 30,
    kde: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    color: str = '#1f77b4',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the distribution of a numeric column.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        column (str): The column to plot
        bins (int, optional): Number of bins for histogram. Defaults to 30.
        kde (bool, optional): Whether to plot KDE. Defaults to True.
        title (Optional[str], optional): Plot title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        color (str, optional): Bar color. Defaults to '#1f77b4'.
        ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
        
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(df[column], bins=bins, kde=kde, color=color, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Distribution of {column}')
    
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(column)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('Frequency')
    
    return ax


def plot_categorical_distribution(
    df: pd.DataFrame, 
    column: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    palette: str = 'viridis',
    ax: Optional[plt.Axes] = None,
    sort_values: bool = True
) -> plt.Axes:
    """
    Plot the distribution of a categorical column.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        column (str): The column to plot
        title (Optional[str], optional): Plot title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        palette (str, optional): Color palette. Defaults to 'viridis'.
        ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
        sort_values (bool, optional): Whether to sort by count. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    value_counts = df[column].value_counts()
    if sort_values:
        value_counts = value_counts.sort_values(ascending=False)
    
    sns.barplot(x=value_counts.index, y=value_counts.values, palette=palette, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Distribution of {column}')
    
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(column)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('Count')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return ax


def plot_scatter(
    df: pd.DataFrame, 
    x_column: str, 
    y_column: str,
    hue_column: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    palette: str = 'viridis',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a scatter plot of two numeric columns.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        x_column (str): Column for x-axis
        y_column (str): Column for y-axis
        hue_column (Optional[str], optional): Column for color coding. Defaults to None.
        title (Optional[str], optional): Plot title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        palette (str, optional): Color palette. Defaults to 'viridis'.
        ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
        
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    sns.scatterplot(
        data=df, 
        x=x_column, 
        y=y_column, 
        hue=hue_column, 
        palette=palette, 
        ax=ax
    )
    
    if title:
        ax.set_title(title)
    else:
        if hue_column:
            ax.set_title(f'{y_column} vs {x_column} by {hue_column}')
        else:
            ax.set_title(f'{y_column} vs {x_column}')
    
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(x_column)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(y_column)
    
    if hue_column:
        plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm',
    annot: bool = True,
    fmt: str = '.2f'
) -> plt.Axes:
    """
    Plot a correlation matrix for numeric columns.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        columns (Optional[List[str]], optional): Columns to include. Defaults to None.
        title (str, optional): Plot title. Defaults to 'Correlation Matrix'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 10).
        cmap (str, optional): Colormap. Defaults to 'coolwarm'.
        annot (bool, optional): Whether to annotate cells. Defaults to True.
        fmt (str, optional): String formatting code. Defaults to '.2f'.
        
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_matrix, 
        annot=annot, 
        cmap=cmap, 
        fmt=fmt, 
        linewidths=0.5
    )
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    return ax


def plot_boxplot(
    df: pd.DataFrame, 
    y_column: str,
    x_column: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    palette: str = 'viridis',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a box plot for a numeric column, optionally grouped by a categorical column.
    
    Args:
        df (pd.DataFrame): The Pokemon dataset
        y_column (str): Numeric column for the box plot
        x_column (Optional[str], optional): Categorical column for grouping. Defaults to None.
        title (Optional[str], optional): Plot title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        palette (str, optional): Color palette. Defaults to 'viridis'.
        ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.
        
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(
        data=df, 
        x=x_column, 
        y=y_column, 
        palette=palette, 
        ax=ax
    )
    
    if title:
        ax.set_title(title)
    else:
        if x_column:
            ax.set_title(f'Distribution of {y_column} by {x_column}')
        else:
            ax.set_title(f'Distribution of {y_column}')
    
    if xlabel and x_column:
        ax.set_xlabel(xlabel)
    elif x_column:
        ax.set_xlabel(x_column)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(y_column)
    
    if x_column:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return ax


def create_subplot_grid(
    n_plots: int,
    n_cols: int = 2,
    figsize: Tuple[int, int] = (16, 12)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots.
    
    Args:
        n_plots (int): Number of plots
        n_cols (int, optional): Number of columns. Defaults to 2.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (16, 12).
        
    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure and array of axes
    """
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Convert axes to 1D array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes