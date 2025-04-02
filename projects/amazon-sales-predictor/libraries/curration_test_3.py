import pandas as pd
import datetime
import numpy as np

def clean_prod(prod_data):
    """Clean and preprocess product data."""
    prod_data = prod_data.drop(columns=['supplier', 'eclipseID'], errors='ignore')
    prod_data = prod_data.rename(columns={
        'SKU': 'sku', 'Brand': 'brand', 'Part Number': 'part_number',
        'Title': 'title', 'Cost+': 'per_unit_cost', 'Type': 'type'
    })
    
    prod_data_cleaned = prod_data[~prod_data['sku'].isna() & (prod_data['sku'].astype(str).str.strip() != 'NaN')]
    prod_data_cleaned.loc[:,'sku'] = prod_data_cleaned['sku'].astype(str)
    prod_data_cleaned = prod_data_cleaned.sort_values(by='sku', key=lambda x: x.str.len(), ascending=False)
    prod_data_cleaned = prod_data_cleaned.drop_duplicates(subset=['brand', 'type', 'title'], keep='first')
    prod_data_cleaned = prod_data_cleaned.drop(columns=['FBA SKUs'], errors='ignore')
    prod_data_cleaned.loc[:,'sku'] = prod_data_cleaned['sku'].str.replace('-', '', regex=False)
    return prod_data_cleaned

def clean_mfn(mfn_):
    """Clean and preprocess MFN data."""
    mfn = mfn_.copy()
    mfn = mfn.rename(columns={
        'OrderID': 'order_id', 'Is Business Order?': 'business_order',
        'Purchase Date': 'purchase_date', 'SKU': 'sku', 'Sell Qty': 'sell_qty',
        'LineItem Total': 'purchase_revenue', 'LineItem Cost': "per_unit_cost",
        'Marketplace Fee': 'marketplace_fee', 'Freight Cost': 'frieght_cost',
        'Total Cost': 'total_purchase_cost', 'Profit': 'purchase_profit'
    })
    
    cols = ['per_unit_cost', 'marketplace_fee', 'frieght_cost']
    mfn[cols] = mfn[cols].abs()
    condition = (mfn['per_unit_cost'].isna()) & (mfn['sell_qty'] == 1) & (~mfn['purchase_revenue'].isna())
    mfn.loc[condition, 'per_unit_cost'] = mfn['purchase_revenue'] + mfn['marketplace_fee'] + mfn['frieght_cost']
    mfn = mfn.dropna(subset=['per_unit_cost'])
    mfn['total_purchase_cost'] = mfn['per_unit_cost'] + mfn['marketplace_fee'] + mfn['frieght_cost']
    mfn['purchase_profit'] = mfn['purchase_revenue'] - mfn['total_purchase_cost']
    return mfn

def clean_fba(fba_sales_):
    # cleans the fba table
    fba_sales = fba_sales_.copy()
    fba_sales = fba_sales_.rename(columns={
        'OrderID': 'order_id', 'Is Business Order?': 'business_order',
        'Purchase Date': 'purchase_date', 'SKU': 'sku', 'Sell Qty': 'sell_qty',
        'LineItem Total': 'purchase_revenue', 'Marketplace Fee': 'marketplace_fee',
        'Fulfillment Fee': 'fulfillment_fee'
    })
    fba_sales['per_unit_cost'] = fba_sales['purchase_revenue'] / fba_sales['sell_qty']
    fba_sales['total_purchase_cost'] = np.abs(fba_sales['marketplace_fee'] + fba_sales['fulfillment_fee'])
    fba_sales['purchase_profit'] = fba_sales['purchase_revenue'] - fba_sales['total_purchase_cost']
    return fba_sales

def filter_table(data):
    """Filter data for valid SKUs and years."""
    valid_skus = data[data['year'] == 2024]['sku'].unique()
    filtered_data = data[data['sku'].isin(valid_skus)]
    sku_year_counts = filtered_data.groupby(['sku', 'year'])['sell_qty'].count().unstack(fill_value=0)
    valid_skus = sku_year_counts[(sku_year_counts[2024] >= 2) & (sku_year_counts.iloc[:, :-1].ge(2).any(axis=1))].index
    return filtered_data[filtered_data['sku'].isin(valid_skus)]

def generate_full_date_range(start_year, end_date):
    """Generate full weekly date ranges from January 1 to the last sale date."""
    start_date = pd.Timestamp(f"{start_year}-01-01")
    return pd.date_range(start=start_date, end=end_date, freq='D')



def create_weekly_date_dataframe(year):
    """
    Creates a DataFrame for the given year with weekly granularity.

    Parameters:
        year (int): The year for which the date range DataFrame is to be created.

    Returns:
        pd.DataFrame: A DataFrame with 'week' and 'week_number' columns.
    """
    # Generate start and end dates for the year
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    
    # Adjust the first week to always start from January 1st
    weekly_dates = [start_date + datetime.timedelta(days=7 * i) for i in range((end_date - start_date).days // 7 + 1)]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'week': pd.to_datetime(weekly_dates),
        'week_number': range(1, len(weekly_dates) + 1)
    })
    
    return df


def full_range_weeks(data, time_frame):
    """Generate full weekly or monthly ranges for each SKU across years."""
    data = data.copy()
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['year']= data['purchase_date'].dt.year
    results = []

    for index ,group in data.groupby(['sku','year']):
        # print(index)
        full_df = create_weekly_date_dataframe(index[1])
        full_df['year'] = full_df['week'].dt.year
        # group['week_number'] = group['purchase_date'].dt.isocalendar().week
        group['month'] = group['purchase_date'].dt.month
       
        start_date = pd.Timestamp(f'{index[1]}-01-01')

        group['week_number'] = ((group['purchase_date'] - start_date).dt.days // 7) + 1

        
        if time_frame == 'weekly':


            group = group.groupby('week_number').agg({'sku':'first',
            'sell_qty': 'sum',
            'business_order': 'sum',
            'purchase_revenue': 'sum',
            'total_purchase_cost': 'sum',
            'purchase_profit': 'sum'
            
            }).reset_index()
            # print(group)
            merged = pd.merge(full_df, group.loc[:,['sku','sell_qty',
            'business_order',
            'purchase_revenue',
            'total_purchase_cost',
            'purchase_profit','week_number']], on=[ 'week_number'], how='left')
            merged = merged.fillna({'sku':index[0],
                           'sell_qty':0,
                           'business_order': 0,
            'purchase_revenue': 0,
            'total_purchase_cost':0,
            'purchase_profit': 0})
            results.append(merged)
            
        # return full_df
        elif time_frame == 'monthly':
            full_df['month'] = full_df['purchase_date'].dt.month
            full_df =  full_df.groupby('month').agg({'year':'first'}).reset_index()
            group = group.groupby('month').agg({'sku':'first',
            'sell_qty': 'sum',
            'business_order': 'sum',
            'purchase_revenue': 'sum',
            'total_purchase_cost': 'sum',
            'purchase_profit': 'sum'
        }).reset_index()
            merged = pd.merge(full_df, group.loc[:,['sku','sell_qty',
            'business_order',
            'purchase_revenue',
            'total_purchase_cost',
            'purchase_profit','month']], on=[ 'month'], how='left')
            merged = merged.fillna({'sku':index[0],
                           'sell_qty':0,
                           'business_order': 0,
            'purchase_revenue': 0,
            'total_purchase_cost':0,
            'purchase_profit': 0})
            results.append(merged)
            
        else:
            results.append(full_df)
            
        
    results = pd.concat(results, ignore_index=True)
    
    return results

def curate_table(prod_data_, mfn_sales_, fba_sales_, time_frame):
    """expected  
    prod_data_: dataframe containing brand, type of each sku
    mfn_sales_: dataframe containing mfn sales
    fba_sales_: dataframe contains fba sales
    time_frame: which granualtiy of data you want, weekly, daily, mounthly
    
    returns all the above tables after cleaning, 
    returns one big dataframe after grouping data to the correct granularity"""
    
    
    
    """Prepare the final dataset by merging and cleaning product, MFN, and FBA data."""
    prod_data = clean_prod(prod_data_)

    mfn_sales = mfn_sales_.rename(columns={
        'OrderID': 'order_id', 'Is Business Order?': 'business_order',
        'Purchase Date': 'purchase_date', 'SKU': 'sku', 'Sell Qty': 'sell_qty',
        'LineItem Total': 'purchase_revenue', 'LineItem Cost': "per_unit_cost",
        'Marketplace Fee': 'marketplace_fee', 'Freight Cost': 'frieght_cost',
        'Total Cost': 'total_purchase_cost', 'Profit': 'purchase_profit'
    })
    mfn_sales = clean_mfn(mfn_sales)

    fba_sales = clean_fba(fba_sales_)
    mfn_sales = mfn_sales.drop(columns=['marketplace_fee', 'frieght_cost'], errors='ignore')
    fba_sales = fba_sales.drop(columns=['marketplace_fee', 'fulfillment_fee'], errors='ignore')

    combined_sales = mfn_sales
    combined_sales['sku'] = combined_sales['sku'].str.replace('-', '', regex=False)
    combined_sales['per_unit_cost'] = combined_sales['per_unit_cost'] / combined_sales['sell_qty']

    
    combined_sales['purchase_date']= pd.to_datetime(combined_sales['purchase_date'])
    combined_sales['year'] = combined_sales['purchase_date'].dt.year
    combined_sales = combined_sales.loc[:,['business_order', 'purchase_date', 'sku', 'sell_qty',
       'purchase_revenue', 'per_unit_cost', 'total_purchase_cost',
       'purchase_profit', 'year']]
    final_df = filter_table(combined_sales)

    if time_frame == 'daily':
       
        final_df = final_df.loc[: ,['purchase_date', 'sku',
            'sell_qty',
            'business_order',
            'per_unit_cost']]
        final_df = final_df.groupby(['sku','purchase_date']).agg({'sell_qty': 'sum',
            'business_order': 'sum',
        }).reset_index()
        
    else:
        final_df = full_range_weeks(final_df, time_frame)

    return prod_data, mfn_sales, fba_sales, combined_sales, final_df

def generate_week_df_for_year(year):
    from datetime import datetime, timedelta
    # Initialize list to hold week data
    week_data = []
    start_date = datetime(year, 1, 1)  # January 1st of the year
    week_number = 1

    # Iterate through the year
    while start_date.year == year:
        week_start = start_date
        week_end = start_date + timedelta(days=6)  # End date is 6 days after start date
        week_data.append((week_start.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d'), week_number))  # Add week start, end, and week number as a tuple
        week_number += 1
        start_date += timedelta(days=7)  # Move to the next week

    # Create a DataFrame from the week_data list
    week_df = pd.DataFrame(week_data, columns=['week', 'week_end', 'week_number'])
    week_df['week'] = pd.to_datetime(week_df['week'])
    week_df['week_end'] = pd.to_datetime(week_df['week_end'])

    return week_df

def generate_combined_week_df_for_years(start_year, end_year):
    # Concatenate weekly data for each year in the given range
    combined_df = pd.DataFrame()  # Empty DataFrame to start
    for year in range(start_year, end_year + 1):
        year_df = generate_week_df_for_year(year)
        combined_df = pd.concat([combined_df, year_df], ignore_index=True)

    return combined_df


def full_range_weeks_test_data(data, time_frame):
    """Generate full weekly or monthly ranges for each SKU across years."""
    data = data.copy()
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['year']= data['purchase_date'].dt.year
    results = []
    start_year = data['year'].min()
    end_year = data['year'].max()
    year_range = generate_combined_week_df_for_years(start_year,end_year)
    
    if time_frame == 'daily':
        data = data.loc[: ,['purchase_date', 'sku',
            'sell_qty',
            'business_order',
            'per_unit_cost']]
        data = data.groupby(['sku','purchase_date']).agg({'sell_qty': 'sum',
            'business_order': 'sum',
            'per_unit_cost':'first',
            
        }).reset_index()
        return data
        
    for index ,group in data.groupby(['sku','year']):
        group = group.reset_index()
        group['month'] = group['purchase_date'].dt.month
        full_df = year_range.copy()
        full_df = full_df[full_df['week']>  pd.to_datetime(f"{index[1]}-{group['month'].min()}-01")]
        full_df['year'] = full_df['week'].dt.year
        start_date = pd.Timestamp(f'{index[1]}-01-01')
        group['week_number'] = ((group['purchase_date'] - start_date).dt.days // 7) + 1

        if time_frame == 'weekly':


            group = group.groupby('week_number').agg({'sku':'first',
                            'sell_qty': 'sum',
                            'business_order': 'sum',
                            'title':'first',
                            'per_unit_cost':'first'
                            }).reset_index()
           
            merged = pd.merge(full_df, group.loc[:,['sku','sell_qty',
                                    'business_order','week_number']],
                                    on=[ 'week_number'], how='left')
            
            merged = merged.fillna({'sku':index[0],
                           'sell_qty':0,
                           'business_order': 0,})
            
            results.append(merged)
            
        # return full_df
        elif time_frame == 'monthly':
            full_df['month'] = full_df['purchase_date'].dt.month
            full_df =  full_df.groupby('month').agg({'year':'first'}).reset_index()
            group = group.groupby('month').agg({'sku':'first',
            'sell_qty': 'sum',
            'business_order': 'sum',
            'purchase_revenue': 'sum',
            'total_purchase_cost': 'sum',
            'purchase_profit': 'sum'
        }).reset_index()
            merged = pd.merge(full_df, group.loc[:,['sku','sell_qty',
            'business_order',
            'purchase_revenue',
            'total_purchase_cost',
            'purchase_profit','month']], on=[ 'month'], how='left')
            merged = merged.fillna({'sku':index[0],
                           'sell_qty':0,
                           'business_order': 0,
            'purchase_revenue': 0,
            'total_purchase_cost':0,
            'purchase_profit': 0})
            results.append(merged)
            
        else: 
            return full_df
            
        
    results = pd.concat(results, ignore_index=True)
    
    return results

def curate_testing_data(data,time_frame):
    import pandas as pd
    from datetime import datetime, timedelta
    df = data.copy()
    df = df.rename(columns={'purchase-date':'purchase_date',
                            'product-name':'title', 
                            'number-of-items':'number_of_items',
                            'quantity':'sell_qty',
                            'item-price':'per_unit_cost',
                            'is-business-order':'business_order'})
    df['per_unit_cost'] =  df['per_unit_cost'] /df['sell_qty']
    df['purchase_date']= pd.to_datetime(df['purchase_date']).dt.date
    df['purchase_date']= pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df = full_range_weeks_test_data(df,time_frame)
    return df 