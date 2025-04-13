"""
Prophet Model Hyperparameter Tuning and Forecasting System

This module provides functionality for:
- Formatting data for Prophet models
- Defining various loss functions
- Hyperparameter tuning using Mango
- Training Prophet models with optimal parameters
- Generating predictions

Dependencies:
- tqdm, prophet, mango, scipy, numpy, tensorflow, pandas
"""

from tqdm import tqdm
from prophet import Prophet
from mango import Tuner
from scipy.stats import uniform
import numpy as np
import tensorflow as tf
import time
import pandas as pd


def format_prophet_data(sku: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Formats training and testing data for Prophet model training for a specific SKU.
    
    Args:
        sku: Stock keeping unit identifier (string)
        train_df: Training dataset containing historical SKU data
        test_df: Testing dataset containing validation SKU data
    
    Returns:
        tuple: (test_data, train_data) formatted DataFrames for Prophet
               (None, None) if error occurs
    
    Processing Steps:
        1. Filters data for the specified SKU
        2. Renames columns to Prophet's expected format ('ds' for dates, 'y' for values)
        3. Adds capacity constraints (cap and floor) for logistic growth
    """
    try:
        # Filter data for specific SKU
        test = test_df[test_df['sku'] == sku][['purchase_date', 'sell_qty']]
        train = train_df[train_df['sku'] == sku][['purchase_date', 'sell_qty']]
        
        # Rename columns to Prophet's expected format
        test = test.rename(columns={'purchase_date': 'ds', 'sell_qty': 'y'})
        train = train.rename(columns={'purchase_date': 'ds', 'sell_qty': 'y'})
        
        # Add capacity constraints for logistic growth
        for df in [train, test]:
            df['cap'] = df['y'].max() + 1  # Upper bound
            df['floor'] = 0  # Lower bound
            
        return test, train
    except Exception as e:
        print(f"Error formatting data for {sku}: {e}")
        return None, None


# --------------------------
# LOSS FUNCTION DEFINITIONS
# --------------------------

def mse(y_true: np.array, y_pred: np.array) -> float:
    """Calculates Mean Squared Error between true and predicted values."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def huber_loss(y_true: np.array, y_pred: np.array, delta: float = 1.0) -> float:
    """
    Calculates Huber loss, which is less sensitive to outliers than MSE.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        delta: Threshold determining transition point between quadratic and linear loss
    
    Returns:
        Calculated Huber loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residual = np.abs(y_true - y_pred)
    loss = np.where(
        residual <= delta,
        0.5 * residual ** 2,  # Quadratic loss for small residuals
        delta * (residual - 0.5 * delta)  # Linear loss for large residuals
    )
    return np.mean(loss)


def poisson_loss(y_true: np.array, y_pred: np.array, epsilon: float = 1e-5) -> float:
    """
    Calculates Poisson loss, suitable for count data.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        epsilon: Small value to prevent division by zero
    
    Returns:
        Calculated Poisson loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred <= 0, epsilon, y_pred)  # Prevent division by zero
    return np.mean(y_pred - y_true * np.log(y_pred))


def mape(y_true: np.array, y_pred: np.array, epsilon: float = 1e-5) -> float:
    """
    Calculates Mean Absolute Percentage Error.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        epsilon: Small value to prevent division by zero when true values are 0
    
    Returns:
        Calculated MAPE
    """
    y_true = np.where(np.array(y_true) == 0, epsilon, np.array(y_true))
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.array, y_pred: np.array, epsilon: float = 1e-5) -> float:
    """
    Calculates Root Mean Squared Error.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        epsilon: Small value to prevent division by zero when true values are 0
    
    Returns:
        Calculated RMSE
    """
    y_true = np.where(np.array(y_true) == 0, epsilon, np.array(y_true))
    return (np.mean((y_true - y_pred)**2))**.5


def custom_weighted_loss(y_actual: np.array, y_pred: np.array, epsilon: float = 1e-3) -> float:
    """
    Custom weighted loss function that gives more weight to values further from the mean.
    
    Args:
        y_actual: Array of true values
        y_pred: Array of predicted values
        epsilon: Small value to ensure non-zero weights
    
    Returns:
        Calculated weighted loss
    """
    # Ensure inputs are TensorFlow tensors
    y_actual = tf.convert_to_tensor(y_actual, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Calculate absolute error
    error = tf.abs(y_actual - y_pred)

    # Calculate weights based on deviation from mean
    mean_actual = tf.reduce_mean(y_actual)
    weights = tf.abs(y_actual - mean_actual) + epsilon

    # Calculate and return weighted loss
    return tf.reduce_mean(weights * error)

loss_functions = {
    'custom': custom_weighted_loss,
    'mse': mse,
    'huber_loss': huber_loss,
    'poisson_loss': poisson_loss,
    'mape': mape,
    'remse': rmse,
    
}

def objective_function(args_list: list, loss_function: callable, test: pd.DataFrame, train: pd.DataFrame) -> tuple:
    """
    Evaluates Prophet model performance for given hyperparameters.
    
    Args:
        args_list: List of parameter dictionaries to evaluate
        loss_function: Loss function to use for evaluation
        test: Testing data DataFrame
        train: Training data DataFrame
    
    Returns:
        tuple: (params_evaluated, results) containing tried parameters and their scores
    """
    params_evaluated = []
    results = []
    
    for params in args_list:
        try:
            Test_size = test.shape[0]
            
            # Initialize and fit Prophet model with current parameters
            model = Prophet(**params)
            model.fit(train)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=Test_size, freq='D')
            forecast = model.predict(future)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)  # Ensure non-negative predictions
            
            # Calculate loss on test set
            predictions_tuned = forecast.tail(Test_size)
            error = loss_function(test['y'], predictions_tuned['yhat'])
            
            # Store results
            params_evaluated.append(params)
            results.append(error)
            
        except Exception as e:
            print(f"Error in objective function evaluation: {e}")
            params_evaluated.append(params)
            results.append(1000.0)  # Large penalty for failed evaluations
            
    return params_evaluated, results


def run_tuning(args: tuple) -> tuple:
    """
    Performs hyperparameter tuning for a Prophet model on a specific SKU.
    
    Args:
        args: Tuple containing:
            - sku: Stock keeping unit identifier
            - testing_data: Test dataset
            - train_data: Training dataset
            - loss_name: Name of loss function being used
            - loss_func: Loss function callable
            - param_space: Parameter search space
            - models: Dictionary to store trained models
    
    Returns:
        tuple: (results DataFrame, updated models dictionary)
    """
    sku, testing_data, train_data, loss_name, param_space, models = args
    
    loss_func = loss_functions[loss_name]
    # Tuner configuration
    TUNER_CONFIG = {
        'initial_random': 10,  # Number of random initial points
        'num_iteration': 30    # Number of optimization iterations
    }
    
    # Format data for Prophet
    test, train = format_prophet_data(sku, train_data, testing_data)
    if test is None or train is None:
        return pd.DataFrame(), models

    # Initialize Mango tuner
    tuner = Tuner(
        param_space,
        lambda params: objective_function(params, loss_func, test, train),
        TUNER_CONFIG
    )
    
    # Execute tuning and measure time
    start_time = time.time()
    results = tuner.minimize()
    training_time = time.time() - start_time

    # Process results into DataFrame
    params_df = pd.DataFrame(list(results['params_tried']))
    params_df['sku'] = sku
    params_df['loss'] = results['objective_values']
    params_df['loss_function'] = loss_name
    params_df['training_time'] = training_time
    
    # Train and store the best model
    best_params = results['best_params']
    model_key = f"{sku}_{loss_name}"
    model = Prophet(**best_params)
    models[model_key] = model.fit(train)
    
    return params_df, models


def train(args_list: list) -> tuple:
    """
    Trains multiple Prophet models with hyperparameter tuning.
    
    Args:
        args_list: List of argument tuples for run_tuning function
    
    Returns:
        tuple: (list of results DataFrames, dictionary of trained models)
    """
    results = []
    # models = {}
    
    # Process each SKU and loss function combination
    for args in tqdm(args_list, desc="Training Models"):
        try:
            result_df, models = run_tuning(args)
            if not result_df.empty:
                results.append(result_df)
        except Exception as e:
            print(f"Error processing {args[0]}-{args[3]}: {e}")
    
    return results, models


def predict(models: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictions using trained Prophet models.
    
    Args:
        models: Dictionary of trained Prophet models
        test_df: Test dataset containing dates and actual values
    
    Returns:
        DataFrame containing predictions with columns:
        ['sku', 'ds', 'yhat', 'sell_qty']
    """
    predictions = []
    
    for model_key, model in models.items():
        try:
            # Extract SKU from model key
            sku = model_key.split('_')[0]
            
            # Prepare data for prediction
            data = test_df[test_df['sku'] == sku].copy()
            data['ds'] = data['purchase_date']
            
            # Generate forecast
            forecast = model.predict(data)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)  # Ensure non-negative predictions
            
            # Store results with actual values
            forecast['sell_qty'] = data['sell_qty'].reset_index(drop=True)
            forecast['sku'] = model_key
            
            # Add to predictions collection
            predictions.append(
                forecast.loc[:, ['sku', 'ds', 'yhat', 'sell_qty']].tail(test_df.shape[0])
            )
            
        except Exception as e:
            print(f"Error generating predictions for {model_key}: {e}")
    
    return pd.concat(predictions)