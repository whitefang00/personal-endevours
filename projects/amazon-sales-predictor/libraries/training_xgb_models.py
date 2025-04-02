import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import curration_test_3 as cu  # Custom module for data cleaning

def process_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Processes sales data by ensuring a continuous date range for each SKU.
    
    This function converts purchase dates to a uniform format, aggregates sales per day,
    and fills in missing dates with zero sales to create a consistent time series.
    
    Parameters:
        table (pd.DataFrame): The input sales data containing SKU, purchase date, and sales quantity.
    
    Returns:
        pd.DataFrame: Processed data with missing dates filled and SKUs aligned.
    """
    data = table.copy()
    data['purchase_date'] = pd.to_datetime(data['purchase_date']).dt.strftime('%Y-%m-%d')
    data = data.groupby(['sku', 'purchase_date']).sum().reset_index()
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    
    start_date, end_date = data['purchase_date'].min(), data['purchase_date'].max()
    res = []
    
    for sku, group in data.groupby('sku'):
        date_range = pd.DataFrame({'purchase_date': pd.date_range(start=start_date, end=end_date)})
        group = group.merge(date_range, on='purchase_date', how='right')
        group.loc[:, 'sku'] = group['sku'].ffill().bfill()  # Fill missing SKU values
        group.fillna(0, inplace=True)  # Fill missing sales with zero
        res.append(group)
    
    return pd.concat(res, ignore_index=True)


def load_data(split_date: str = '2024-01-01', create_test_train_split=True):
    """
    Loads and preprocesses the training and validation datasets.
    
    Reads sales and product data, filters SKUs present in both sets, processes the sales table,
    and optionally splits the dataset into training and testing subsets.
    
    Parameters:
        split_date (str): Date to split the dataset into training and testing.
        create_test_train_split (bool): Whether to return a split dataset or a full dataset.
    
    Returns:
        Tuple: (train_df, test_df, prod, sku_list) if split; otherwise (df, prod, sku_list).
    """
    df = pd.read_csv('data/currated/train_data_daily_20211001_20241025.csv').drop(columns='Unnamed: 0')
    prod = cu.clean_prod(pd.read_csv('data/raw/prod.csv'))
    validate_df = pd.read_csv('data/currated/testing_data_daily_currated_20241015_20241215.csv').drop(['Unnamed: 0','cap','floor','per_unit_cost'],axis =1)
    
    validate_skus = validate_df['sku'].unique()
    prod_sku = prod['sku'].unique()
    current_skus = df[(df['sku'].isin(validate_skus)) & (df['sku'].isin(prod_sku))]['sku'].unique()
    
    # Identify SKUs with highest sales in both training and validation datasets
    sku_1 = validate_df[validate_df['sku'].isin(current_skus)].groupby('sku').agg({'sell_qty': 'sum'})
    sku_2 = df[df['sku'].isin(current_skus)].groupby('sku').agg({'sell_qty': 'sum'})
    sku_list = sku_1.nlargest(20, 'sell_qty').index.intersection(sku_2.nlargest(20, 'sell_qty').index)
    
    df = process_table(df).merge(prod, on='sku', how='left')
    
    if create_test_train_split:
        train_df = df[df['purchase_date'] < split_date]
        test_df = df[df['purchase_date'] >= split_date]
        return train_df, test_df, prod, sku_list
    else:
        test_df = process_table(validate_df).merge(prod, on='sku',how ='left')
        return df,test_df, prod, sku_list


def plot_predictions(train_df: pd.DataFrame, predictions: pd.DataFrame, model_type: str):
    """
    Plots actual vs. predicted sales quantity for different SKUs.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset.
        predictions (pd.DataFrame): Model predictions.
        model_type (str): Type of model used (e.g., 'XGBoost', 'Random Forest').
    """
    for sku in predictions['sku'].unique():
        x = predictions[predictions['sku'] == sku]
        training_sku = train_df[train_df['sku'] == sku][['purchase_date', 'sell_qty']].tail(50)
        
        plt.figure(figsize=(20, 6))
        plt.plot(x['purchase_date'], x['y_hat'], label='Predicted Sell Qty', linewidth=2, color='blue')
        plt.plot(x['purchase_date'], x['sell_qty'], label='Actual Sell Qty', linewidth=2, color='red')
        plt.plot(training_sku['purchase_date'], training_sku['sell_qty'], label='Recent Trend', linestyle='--', linewidth=2, color='orange')
        
        plt.xlabel('Date')
        plt.ylabel('Sell Quantity')
        plt.title(f'Actual vs Predicted Sales for SKU: {sku}, Model: {model_type}')
        plt.legend()
        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.show()


def train_models_advanced(train_data: pd.DataFrame, type: str, price: float, test_data: pd.DataFrame = None, split: bool = True):
    """
    Trains an XGBoost regression model for sales prediction.
    
    Parameters:
        train_data (pd.DataFrame): Training dataset.
        type (str): Product type to filter training data.
        price (float): Price range filter.
        test_data (pd.DataFrame): Test dataset (if applicable).
        split (bool): Whether to split data into training/testing sets.
    
    Returns:
        xgb.Booster: Trained XGBoost model.
    """
    # Filtering data by product type and price range
    train_data = train_data[(train_data['type'] == type) & 
                            (train_data['per_unit_cost'].between(price - 10, price + 10))]
    
    # Creating time-based features
    train_data['purchase_date'] = pd.to_datetime(train_data['purchase_date'])
    train_data['day'] = train_data['purchase_date'].dt.day
    train_data['month'] = train_data['purchase_date'].dt.month
    train_data['year'] = train_data['purchase_date'].dt.year
    
    # Encoding categorical variables
    label_encoders = {col: LabelEncoder().fit(train_data[col]) for col in ['brand', 'type', 'part_number', 'title']}
    for col, le in label_encoders.items():
        train_data[col] = le.transform(train_data[col])
    
    # Creating lag features for time-series forecasting
    for lag in [1, 7, 14, 30]:
        train_data[f'sell_qty_lag_{lag}'] = train_data.groupby('sku')['sell_qty'].shift(lag)
    train_data['sell_qty_rolling_mean_7'] = train_data.groupby('sku')['sell_qty'].rolling(window=7).mean().reset_index(0, drop=True)
    train_data['sell_qty_rolling_std_7'] = train_data.groupby('sku')['sell_qty'].rolling(window=7).std().reset_index(0, drop=True)

    train_data.fillna(0, inplace=True)  # Handling missing values
    
    # Splitting the dataset
    X = train_data.drop(['sku', 'purchase_date', 'sell_qty'], axis=1)
    y = train_data['sell_qty']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    dtrain, dtest = xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_test, label=y_test)
    
    # Training the XGBoost model
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.1, 'max_depth': 6}
    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=50)
    
    return model

def predict_future_advanced(test_data: pd.DataFrame, models: dict, history_data: pd.DataFrame,test:bool):
    """
    Predict future sales using trained XGBoost models.

    Parameters:
        test_data (pd.DataFrame): The dataset for which predictions are required.
        models (dict): Dictionary mapping SKU to trained XGBoost models.
        history_data (pd.DataFrame): Historical sales data to compute lag features.

    Returns:
        pd.DataFrame: Predictions with SKU, purchase_date, and predicted sell_qty.
    """
    test_data = test_data.copy()
    test_data['purchase_date'] = pd.to_datetime(test_data['purchase_date'])
    test_data['day'] = test_data['purchase_date'].dt.day
    test_data['month'] = test_data['purchase_date'].dt.month
    test_data['year'] = test_data['purchase_date'].dt.year
    # print(test_data.columns)
    label_encoders = {col: LabelEncoder().fit(test_data[col]) for col in ['brand', 'type', 'part_number', 'title']}
    for col, le in label_encoders.items():
        test_data[col] = le.transform(test_data[col])
    
    # test_data['sku_float'] = test_data['sku'].astype('category').cat.codes.astype(float)

    results = []
    
    for sku, group in test_data.groupby('sku'):

        if sku not in models.keys():
            continue  # Skip SKUs without trained models
        
        model = models[sku]
        past_sales = history_data[history_data['sku'] == sku].copy()
        
        # Sort data to maintain chronological order
        past_sales = past_sales.sort_values('purchase_date')
        group = group.sort_values('purchase_date')
        # print(group.shape)
        for lag in [1, 7, 14, 30]:
            group[f'sell_qty_lag_{lag}'] = past_sales['sell_qty'].shift(lag).iloc[-len(group):].values
        
        group['sell_qty_rolling_mean_7'] = past_sales['sell_qty'].rolling(window=7).mean().iloc[-len(group):].values
        group['sell_qty_rolling_std_7'] = past_sales['sell_qty'].rolling(window=7).std().iloc[-len(group):].values

        # Handle missing values
        lag_cols = [f'sell_qty_lag_{lag}' for lag in [1, 7, 14, 30]]
        group[lag_cols] = group[lag_cols].fillna(0)  # Or use mean
        
        group[['sell_qty_rolling_mean_7', 'sell_qty_rolling_std_7']] = group[['sell_qty_rolling_mean_7', 'sell_qty_rolling_std_7']].fillna(0)

        X_test = group.drop(['sku', 'purchase_date', 'sell_qty'], axis=1, errors='ignore')
        dtest = xgb.DMatrix(X_test)
        group['y_hat'] = model.predict(dtest)
        if test:
            results.append(group[['sku', 'purchase_date', 'sell_qty','y_hat']])
        else:
            results.append(group[['sku', 'purchase_date', 'y_hat']])
    return pd.concat(results, ignore_index=True)