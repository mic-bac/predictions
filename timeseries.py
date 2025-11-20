# %% 
# ============================================================================
# WALMART SALES FORECASTING: PROPHET & XGBOOST TUTORIAL
# ============================================================================
# 
# This notebook demonstrates time series forecasting techniques for 
# predicting Walmart store sales using:
# - Facebook Prophet: Automated time series model with seasonality
# - XGBoost: Gradient boosting for feature-based predictions
# - Scikit-learn: Baseline linear regression model
#
# Dataset: Weekly sales data for 45 Walmart stores (2010-2013)
# Students: This course assumes basic statistics knowledge
# ============================================================================

# %% 
# IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Prophet and XGBoost
from prophet import Prophet
import xgboost as xgb

# For time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Install plotting function
from src.utils.ts_plot import plot_time_series

# %%
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
# The Walmart dataset consists of 4 CSV files:
# - train.csv: Historical sales data with target variable
# - test.csv: Test set (without sales values)
# - stores.csv: Store metadata (type and size)
# - features.csv: External features (temperature, fuel price, markdowns, etc.)

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('./data/sales/train.csv')
stores_df = pd.read_csv('./data/sales/stores.csv')
features_df = pd.read_csv('./data/sales/features.csv')



print(f"Train shape: {train_df.shape}")
print(f"Stores shape: {stores_df.shape}")
print(f"Features shape: {features_df.shape}")

# Display sample data
print("\n--- Sample Training Data ---")
display(train_df.head())
print("\n--- Train Data Info ---")
print(train_df.info())
print("\n--- Train Data Statistics ---")
display(train_df.describe())

print("\n--- Stores Data ---")
display(stores_df.head())

print("\n--- Features Data (First Few Rows) ---")
display(features_df.head())

# %%
# 2. DATA PREPARATION AND MERGING
# ============================================================================
# Merge all datasets to create a comprehensive feature set

print("\nPreparing data...")

# Convert Date column to datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])
features_df['Date'] = pd.to_datetime(features_df['Date'])

features_df.drop("IsHoliday", axis=1, inplace=True) # otherwise we have it doubled

# Merge training data with stores information
train_merged = train_df.merge(stores_df, on='Store', how='left')

# Merge with features
train_merged = train_merged.merge(features_df, on=['Store', 'Date'], how='left')

# Remove all negative sales
train_merged = train_merged[train_merged["Weekly_Sales"] > 0]

print(f"Merged train shape: {train_merged.shape}")
print("\n--- Combined Data Info ---")
print(train_merged.info())

# %%
# 3. MISSING VALUES ANALYSIS AND HANDLING
# ============================================================================
# Understanding and handling missing values is critical for forecasting

print("\n--- Missing Values Analysis ---")
missing_summary = pd.DataFrame({
    'Column': train_merged.columns,
    'Missing_Count': train_merged.isnull().sum(),
    'Missing_Percentage': (train_merged.isnull().sum() / len(train_merged) * 100).round(2)
})
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Handle missing values:
# 1. MarkDown columns: Missing means no markdown event (fill with 0)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(0)

# 2. CPI and Unemployment: Use forward fill then backfill for time series
if 'CPI' in train_merged.columns:
    train_merged['CPI'] = train_merged.groupby('Store')['CPI'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )

if 'Unemployment' in train_merged.columns:
    train_merged['Unemployment'] = train_merged.groupby('Store')['Unemployment'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )

print("\n--- Missing values after handling ---")
print(f"Train missing values: {train_merged.isnull().sum().sum()}")

# %%
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
# Visualizing key patterns and distributions in the data

# aggregate on store level to make visual more meaningful
train_merged_eda = train_merged.groupby(["Store", "Date", "IsHoliday", "Type"])["Weekly_Sales"].sum().reset_index()

# 4.1: Distribution of weekly sales
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train_merged_eda['Weekly_Sales'],
    nbinsx=50,
    name='Weekly Sales',
    marker_color='royalblue'
))
fig.update_layout(
    title='Distribution of Weekly Sales',
    xaxis_title='Weekly Sales',
    yaxis_title='Frequency',
    height=500,
    template='plotly_white'
)
fig.show()

# 4.2: Sales trend over time (aggregated)
sales_by_date = train_merged_eda.groupby('Date')['Weekly_Sales'].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sales_by_date['Date'],
    y=sales_by_date['Weekly_Sales'],
    mode='lines',
    name='Total Weekly Sales',
    line=dict(color='green', width=2)
))
fig.update_layout(
    title='Total Weekly Sales Over Time',
    xaxis_title='Date',
    yaxis_title='Sales',
    height=500,
    template='plotly_white',
    hovermode='x unified'
)
fig.show()

# 4.3: Holiday impact on sales
holiday_impact = train_merged_eda.groupby('IsHoliday')['Weekly_Sales'].agg(['mean', 'median', 'std'])
print("\n--- Holiday Impact on Sales ---")
print(holiday_impact)

fig = go.Figure()
for is_holiday in [True, False]:
    data = train_merged_eda[train_merged_eda['IsHoliday'] == is_holiday]['Weekly_Sales']
    fig.add_trace(go.Box(
        y=data,
        name='Holiday' if is_holiday else 'Non-Holiday',
        boxmean='sd'
    ))
fig.update_layout(
    title='Sales Distribution: Holiday vs Non-Holiday Weeks',
    yaxis_title='Weekly Sales',
    height=500,
    template='plotly_white'
)
fig.show()

# 4.4: Store type analysis
store_type_sales = train_merged_eda.groupby('Type')['Weekly_Sales'].agg(['mean', 'median'])
print("\n--- Sales by Store Type ---")
print(store_type_sales)

fig = px.box(train_merged_eda, x='Type', y='Weekly_Sales', 
             title='Weekly Sales Distribution by Store Type',
             labels={'Type': 'Store Type', 'Weekly_Sales': 'Sales'})
fig.update_layout(height=500, template='plotly_white')
fig.show()

# %%
# 5. TIME SERIES DECOMPOSITION
# ============================================================================
# Understanding trend, seasonality, and residuals helps us choose models

print("\nPerforming time series decomposition...")

# For decomposition, we'll use a single store aggregated data
# (decomposition requires no missing values and regular frequency)
store_1_sales = train_merged[train_merged['Store'] == 1].sort_values('Date')
store_1_sales_agg = store_1_sales.groupby('Date')['Weekly_Sales'].sum()

# Perform seasonal decomposition
decomposition = seasonal_decompose(store_1_sales_agg, model='additive', period=52)

fig = px.scatter(x=pd.DataFrame(store_1_sales_agg).index, y=store_1_sales_agg, trendline="ols")
fig.show()
decomposition.plot()

print("Decomposition complete. Key observations:")
print("- Trend: Long-term direction of sales")
print("- Seasonality: Regular patterns (e.g., yearly spikes in holidays)")
print("- Residuals: Random noise after removing trend and seasonality")

# %%
# 6. FEATURE ENGINEERING
# ============================================================================
# Creating new features from existing data to improve model performance

def create_features(df):
    """
    Engineer time-based and domain-specific features
    """
    df_copy = df.copy()
    
    # Extract time components
    df_copy['Year'] = df_copy['Date'].dt.year
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['Week'] = df_copy['Date'].dt.isocalendar().week
    df_copy['Quarter'] = df_copy['Date'].dt.quarter
    
    # Encode categorical variables
    le_type = LabelEncoder()
    df_copy['Type_Encoded'] = le_type.fit_transform(df_copy['Type'])
    
    df_copy['IsHoliday_Int'] = df_copy['IsHoliday'].astype(int)
    
    # Create interaction features
    df_copy['Size_Type'] = df_copy['Size'] * df_copy['Type_Encoded']
    
    # Create cyclical features for seasonality (month and week)
    # These help models understand that December and January are similar cyclically
    df_copy['Month_Sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
    df_copy['Month_Cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)
    df_copy['Week_Sin'] = np.sin(2 * np.pi * df_copy['Week'] / 52)
    df_copy['Week_Cos'] = np.cos(2 * np.pi * df_copy['Week'] / 52)
    
    return df_copy

train_engineered = create_features(train_merged)

print("Features created:")
print(train_engineered[['Date', 'Year', 'Month', 'Week', 
                         'Month_Sin', 'Month_Cos']].head())


# We create a test and a training set (for TS we just take the last 5% of dates available for test)
train_engineered.drop(markdown_cols, axis=1, inplace=True) # we drop them as we shouldn't know them in the future
df_plot_y = train_engineered[["Date", "Store", "Dept", "Weekly_Sales"]]
train_dates, test_dates = np.split(train_engineered.Date.unique(), [int(.95 *len(train_engineered.Date.unique()))])
test_engineered = train_engineered[train_engineered.Date.isin(test_dates)]
train_engineered = train_engineered[train_engineered.Date.isin(train_dates)]


# %%
# 7. BASELINE MODEL: LINEAR REGRESSION (with scikit-learn)
# ============================================================================
# A simple baseline to establish a performance benchmark

print("\n" + "="*70)
print("BASELINE MODEL: LINEAR REGRESSION")
print("="*70)

# Select features for the baseline model
feature_cols = ['Store', 'Dept', 'Size', 'Type_Encoded', 'IsHoliday_Int',
                'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Quarter',
                'Month', 'Week', 'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos']

# Select final train and test values
baseline_train = train_engineered.dropna(subset=feature_cols + ['Weekly_Sales'])
baseline_test = test_engineered.dropna(subset=feature_cols)

X_train_baseline = baseline_train[feature_cols].copy()
y_train_baseline = baseline_train['Weekly_Sales']

X_test_baseline = baseline_test[feature_cols].copy()
y_test_baseline = baseline_test['Weekly_Sales']

# IMPORTANT: Scale features for Linear Regression
# Create scaler on training data
lr_scaler = StandardScaler()
X_train_baseline_scaled = lr_scaler.fit_transform(X_train_baseline)
X_train_baseline_scaled = pd.DataFrame(X_train_baseline_scaled, columns=feature_cols)

X_test_baseline_scaled = lr_scaler.transform(X_test_baseline)
X_test_baseline_scaled = pd.DataFrame(X_test_baseline_scaled, columns=feature_cols)

print("\nLinear Regression uses SCALED features:")
print(f"  Mean of scaled features: {X_train_baseline_scaled.mean().mean():.6f}")
print(f"  Std of scaled features: {X_train_baseline_scaled.std().mean():.6f}")

# Train linear regression on scaled data
lr_model = LinearRegression()
lr_model.fit(X_train_baseline_scaled, y_train_baseline)

# Predictions on training set (for evaluation)
y_train_pred_lr = lr_model.predict(X_train_baseline_scaled)
y_test_pred_lr = lr_model.predict(X_test_baseline_scaled)

# Calculate metrics
mae_lr = mean_absolute_error(y_train_baseline, y_train_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_train_baseline, y_train_pred_lr))
r2_lr = r2_score(y_train_baseline, y_train_pred_lr)

mae_lr_test = mean_absolute_error(y_test_baseline, y_test_pred_lr)
rmse_lr_test = np.sqrt(mean_squared_error(y_test_baseline, y_test_pred_lr))
r2_lr_test = r2_score(y_test_baseline, y_test_pred_lr)

print(f"\nLinear Regression - Training Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_lr:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_lr:,.2f}")
print(f"  R² Score: {r2_lr:.4f}")

print(f"\nLinear Regression - Test Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_lr_test:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_lr_test:,.2f}")
print(f"  R² Score: {r2_lr_test:.4f}")

# Feature importance for linear regression
feature_importance_lr = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features (by coefficient):")
print(feature_importance_lr.head(10))

fig = go.Figure(data=[
    go.Bar(x=feature_importance_lr.head(10)['Coefficient'].values,
           y=feature_importance_lr.head(10)['Feature'].values,
           orientation='h')
])
fig.update_layout(
    title='Linear Regression: Top 10 Feature Coefficients',
    xaxis_title='Coefficient Value',
    yaxis_title='Feature',
    height=500,
    template='plotly_white'
)
fig.show()


# Plot result for one store
df_plot_y["yhat"] = np.concatenate([y_train_pred_lr, y_test_pred_lr])
df_plot_ygrp = df_plot_y.groupby(["Date", "Store"]).agg({"Weekly_Sales": "sum", "yhat": "sum"}).reset_index()
df_plot_y.drop('yhat', axis=1, inplace=True)

df_plot_ygrp[df_plot_ygrp.Store == 1]

plot_time_series(
    x=df_plot_ygrp[df_plot_ygrp.Store == 1].Date.values, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales.values, 
    yhat=df_plot_ygrp[df_plot_ygrp.Store == 1].yhat.values,
    vline_x=train_dates.max(),
    title="Linear Regression Stores and Dpt"
)

# %% 7.1 Linear regression on store level
grp_feature_cols = feature_cols.copy()
grp_feature_cols.pop(1)
df_agg_store_train = baseline_train.groupby(grp_feature_cols + ["Date"]).agg({"Weekly_Sales": "sum"}).reset_index()
df_agg_store_test = baseline_test.groupby(grp_feature_cols + ["Date"]).agg({"Weekly_Sales": "sum"}).reset_index()

X_train_baseline = df_agg_store_train[grp_feature_cols].copy()
y_train_baseline = df_agg_store_train['Weekly_Sales']

X_test_baseline = df_agg_store_test[grp_feature_cols].copy()
y_test_baseline = df_agg_store_test['Weekly_Sales']

lr_scaler = StandardScaler()
X_train_baseline_scaled = lr_scaler.fit_transform(X_train_baseline)
X_train_baseline_scaled = pd.DataFrame(X_train_baseline_scaled, columns=grp_feature_cols)
X_test_baseline_scaled = lr_scaler.transform(X_test_baseline)
X_test_baseline_scaled = pd.DataFrame(X_test_baseline_scaled, columns=grp_feature_cols)
# Train linear regression on scaled data
lr_model = LinearRegression()
lr_model.fit(X_train_baseline_scaled, y_train_baseline)

# Predictions on training set (for evaluation)
y_train_pred_lr = lr_model.predict(X_train_baseline_scaled)
y_test_pred_lr = lr_model.predict(X_test_baseline_scaled)

# Calculate metrics
mae_lr = mean_absolute_error(y_train_baseline, y_train_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_train_baseline, y_train_pred_lr))
r2_lr = r2_score(y_train_baseline, y_train_pred_lr)

mae_lr_test = mean_absolute_error(y_test_baseline, y_test_pred_lr)
rmse_lr_test = np.sqrt(mean_squared_error(y_test_baseline, y_test_pred_lr))
r2_lr_test = r2_score(y_test_baseline, y_test_pred_lr)

print(f"\nLinear Regression - Training Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_lr:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_lr:,.2f}")
print(f"  R² Score: {r2_lr:.4f}")

print(f"\nLinear Regression - Test Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_lr_test:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_lr_test:,.2f}")
print(f"  R² Score: {r2_lr_test:.4f}")

# Plot result for one store
df_agg_store_train["yhat"] = y_train_pred_lr
df_agg_store_test["yhat"] = y_test_pred_lr
df_plot_ygrp = pd.concat([df_agg_store_train, df_agg_store_test], axis=0)
df_plot_ygrp = df_plot_ygrp.sort_values("Date").reset_index(drop=True)

df_plot_ygrp[df_plot_ygrp.Store == 1]

plot_time_series(
    x=df_plot_ygrp[df_plot_ygrp.Store == 1].Date.values, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales.values, 
    yhat=df_plot_ygrp[df_plot_ygrp.Store == 1].yhat.values,
    vline_x=train_dates.max(),
    title="Linear Regression all Stores"
)

# %% 7.2 Linear Regression on Store 1 only
X_train_baseline = df_agg_store_train[grp_feature_cols][df_agg_store_train.Store == 1].copy()
y_train_baseline = df_agg_store_train[df_agg_store_train.Store == 1]['Weekly_Sales']

X_test_baseline = df_agg_store_test[grp_feature_cols][df_agg_store_test.Store == 1].copy()
y_test_baseline = df_agg_store_test[df_agg_store_test.Store == 1]['Weekly_Sales']

lr_scaler = StandardScaler()
X_train_baseline_scaled = lr_scaler.fit_transform(X_train_baseline)
X_train_baseline_scaled = pd.DataFrame(X_train_baseline_scaled, columns=grp_feature_cols)
X_test_baseline_scaled = lr_scaler.transform(X_test_baseline)
X_test_baseline_scaled = pd.DataFrame(X_test_baseline_scaled, columns=grp_feature_cols)
# Train linear regression on scaled data
lr_model = LinearRegression()
lr_model.fit(X_train_baseline_scaled, y_train_baseline)

# Predictions on training set (for evaluation)
y_train_pred_lr = lr_model.predict(X_train_baseline_scaled)
y_test_pred_lr = lr_model.predict(X_test_baseline_scaled)

# Calculate metrics
mae_lr = mean_absolute_error(y_train_baseline, y_train_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_train_baseline, y_train_pred_lr))
r2_lr = r2_score(y_train_baseline, y_train_pred_lr)

mae_lr_test = mean_absolute_error(y_test_baseline, y_test_pred_lr)
rmse_lr_test = np.sqrt(mean_squared_error(y_test_baseline, y_test_pred_lr))
r2_lr_test = r2_score(y_test_baseline, y_test_pred_lr)

print(f"\nLinear Regression - Training Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_lr:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_lr:,.2f}")
print(f"  R² Score: {r2_lr:.4f}")

print(f"\nLinear Regression - Test Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_lr_test:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_lr_test:,.2f}")
print(f"  R² Score: {r2_lr_test:.4f}")

# Plot result for one store
df_agg_store_train_1 = df_agg_store_train[df_agg_store_train.Store == 1].reset_index()
df_agg_store_test_1 = df_agg_store_test[df_agg_store_test.Store == 1].reset_index()

df_agg_store_train_1["yhat"] = y_train_pred_lr
df_agg_store_test_1["yhat"] = y_test_pred_lr
df_plot_ygrp = pd.concat([df_agg_store_train_1, df_agg_store_test_1], axis=0)
df_plot_ygrp = df_plot_ygrp.sort_values("Date").reset_index(drop=True)

df_plot_ygrp[df_plot_ygrp.Store == 1]

plot_time_series(
    x=df_plot_ygrp[df_plot_ygrp.Store == 1].Date.values, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales.values, 
    yhat=df_plot_ygrp[df_plot_ygrp.Store == 1].yhat.values,
    vline_x=train_dates.max(),
    title="Linear Regression Store 1"
)


# %%
# 8. PROPHET MODEL: TIME SERIES FORECASTING for Store 1 only
# ============================================================================
# Facebook Prophet is designed for time series with seasonality and trends

print("\n" + "="*70)
print("TIME SERIES MODEL: FACEBOOK PROPHET")
print("="*70)

# Prepare data for Prophet (it requires 'ds' and 'y' columns)
# We'll train on aggregated store data for simplicity in demonstration

prophet_train = baseline_train.copy()
prophet_test = baseline_test.copy()
prophet_test = prophet_test.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
#prophet_test["y"] = np.nan

# Rename columns for Prophet
prophet_df = prophet_train.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
prophet_predict = pd.concat([prophet_df, prophet_test], axis=0).reset_index(drop=True)

# Aggregate on Store Level
prophet_df = prophet_df.groupby(
    [
        'Store', 'ds', 'Size', 'Temperature',
        'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week', 'Quarter',
        'Type_Encoded', 'IsHoliday_Int', 'Size_Type', 'Month_Sin', 'Month_Cos',
        'Week_Sin', 'Week_Cos'
    ]
).agg({"y": "sum"}).reset_index()

prophet_predict = prophet_predict.groupby(
    [
        'Store', 'ds', 'Size', 'Temperature',
        'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week', 'Quarter',
        'Type_Encoded', 'IsHoliday_Int', 'Size_Type', 'Month_Sin', 'Month_Cos',
        'Week_Sin', 'Week_Cos'
    ]
).agg({"y": "sum"}).reset_index()

# Identify holidays (these are the major Walmart holidays)
walmart_holidays = pd.DataFrame({
    'holiday': np.repeat(['Superbowl', 'Labor Day', 'Thanksgiving', 'Christmas'], 4),
    'ds': pd.to_datetime([
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
        '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06',
        '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29',
        '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'
    ]),
    'lower_window': -3,
    'upper_window': 2,
})

# Initialize and fit Prophet model
print("\nTraining Prophet model...")
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    interval_width=0.95,
    seasonality_mode='additive',
    holidays=walmart_holidays
)

# Add external regressors (optional, improves accuracy)
for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
            'IsHoliday_Int', 'Year', 'Month', 'Quarter', 
            'Type_Encoded', 'IsHoliday_Int']:
    prophet_model.add_regressor(col)

# Add holiday effects
prophet_model.add_country_holidays(country_name='US')

# Fit on Store 1
prophet_model.fit(prophet_df[prophet_df.Store == 1])
print("Prophet model training complete!")


# Predict prophet

prophet_yhat = prophet_predict[prophet_predict["Store"] == 1].copy()
prophet_yhat["yhat"] = float()
prophet_yhat["yhat_lower"] = float()
prophet_yhat["yhat_upper"] = float()

for store in [1]:
    print(f"Started store {store}")
    forecast_prophet = prophet_model.predict(prophet_predict[prophet_predict["Store"] == store])

    prophet_yhat.loc[prophet_yhat["Store"] == store, "yhat"] = forecast_prophet["yhat"].values
    prophet_yhat.loc[prophet_yhat["Store"] == store, "yhat_lower"] = forecast_prophet["yhat_lower"].values
    prophet_yhat.loc[prophet_yhat["Store"] == store, "yhat_upper"] = forecast_prophet["yhat_upper"].values

print("\nProphet Forecast Test:")
print(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


plot_time_series(
    x=forecast_prophet.ds, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales, 
    yhat=forecast_prophet.yhat,
    yhat_lower=forecast_prophet.yhat_lower,
    yhat_upper=forecast_prophet.yhat_lower,
    vline_x=train_dates.max(),
    title="Prophet for Store 1"
)

# Plot the forecast
fig = prophet_model.plot(forecast_prophet, figsize=(14, 8))
fig.suptitle('Prophet: Sales Forecast with Confidence Intervals', fontsize=14, y=0.98)
fig.show()

# Plot components (trend, seasonality, holidays)
fig_components = prophet_model.plot_components(forecast_prophet, figsize=(14, 10))
fig_components.suptitle('Prophet: Trend, Seasonality, and Holiday Components', 
                         fontsize=14, y=0.995)
fig_components.show()

# Calculate metrics
y_true = prophet_yhat[prophet_yhat.ds < baseline_test.Date.min()].y
y_hat = prophet_yhat[prophet_yhat.ds < baseline_test.Date.min()].yhat
mae_pr = mean_absolute_error(y_true, y_hat)
rmse_pr = np.sqrt(mean_squared_error(y_true, y_hat))
r2_pr = r2_score(y_true, y_hat)

y_true = prophet_yhat[prophet_yhat.ds >= baseline_test.Date.min()].y
y_hat = prophet_yhat[prophet_yhat.ds >= baseline_test.Date.min()].yhat
mae_pr_test = mean_absolute_error(y_true, y_hat)
rmse_pr_test = np.sqrt(mean_squared_error(y_true, y_hat))
r2_pr_test = r2_score(y_true, y_hat)

print(f"\nProphet - Training Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_pr:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_pr:,.2f}")
print(f"  R² Score: {r2_pr:.4f}")

print(f"\nProphet - Test Set Performance:")
print(f"  Mean Absolute Error (MAE): ${mae_pr_test:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse_pr_test:,.2f}")
print(f"  R² Score: {r2_pr_test:.4f}")

# %%
# 9. XGBOOST MODEL: GRADIENT BOOSTING REGRESSION
# ============================================================================
# XGBoost is a powerful tree-based ensemble method for structured data

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    baseline_train[feature_cols], baseline_train.Weekly_Sales, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: {X_train_split.shape}")
print(f"Validation set shape: {X_val_split.shape}")

# Initialize XGBoost model with optimized hyperparameters
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train with early stopping
xgb_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

print("XGBoost model training complete!")

# Make predictions
y_train_pred_xgb = xgb_model.predict(X_train_split)
y_val_pred_xgb = xgb_model.predict(X_val_split)
y_test_pred_xgb = xgb_model.predict(baseline_test[feature_cols])

train_split_yhat = X_train_split.join(baseline_train["Date"])
train_split_yhat["yhat"] = y_train_pred_xgb
train_split_yhat["Weekly_Sales"] = y_train_split
val_split_yhat = X_val_split.join(baseline_train["Date"])
val_split_yhat["yhat"] = y_val_pred_xgb
val_split_yhat["Weekly_Sales"] = y_val_split
test_split_yhat = baseline_test.copy()
test_split_yhat["yhat"] = y_test_pred_xgb


df_plot_y = pd.concat(
    [
        train_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]], 
        val_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]], 
        test_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]]
    ]
).reset_index(drop=True)
df_plot_ygrp = df_plot_y.groupby(["Date", "Store"]).agg({"Weekly_Sales": "sum", "yhat": "sum"}).sort_values("Date").reset_index()
df_plot_y.drop('yhat', axis=1, inplace=True)

df_plot_ygrp[df_plot_ygrp.Store == 1]

plot_time_series(
    x=df_plot_ygrp[df_plot_ygrp.Store == 1].Date.values, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales.values, 
    yhat=df_plot_ygrp[df_plot_ygrp.Store == 1].yhat.values,
    vline_x=train_dates.max(),
    title="XGBoost for all Stores and Dept"
)



# Calculate metrics
mae_xgb_train = mean_absolute_error(y_train_split, y_train_pred_xgb)
rmse_xgb_train = np.sqrt(mean_squared_error(y_train_split, y_train_pred_xgb))
r2_xgb_train = r2_score(y_train_split, y_train_pred_xgb)

mae_xgb_val = mean_absolute_error(y_val_split, y_val_pred_xgb)
rmse_xgb_val = np.sqrt(mean_squared_error(y_val_split, y_val_pred_xgb))
r2_xgb_val = r2_score(y_val_split, y_val_pred_xgb)

mae_xgb_test = mean_absolute_error(baseline_test.Weekly_Sales, y_test_pred_xgb)
rmse_xgb_test = np.sqrt(mean_squared_error(baseline_test.Weekly_Sales, y_test_pred_xgb))
r2_xgb_test = r2_score(baseline_test.Weekly_Sales, y_test_pred_xgb)


print(f"\nXGBoost - Training Set Performance:")
print(f"  MAE: ${mae_xgb_train:,.2f}")
print(f"  RMSE: ${rmse_xgb_train:,.2f}")
print(f"  R² Score: {r2_xgb_train:.4f}")

print(f"\nXGBoost - Validation Set Performance:")
print(f"  MAE: ${mae_xgb_val:,.2f}")
print(f"  RMSE: ${rmse_xgb_val:,.2f}")
print(f"  R² Score: {r2_xgb_val:.4f}")

print(f"\nXGBoost - Test Set Performance:")
print(f"  MAE: ${mae_xgb_test:,.2f}")
print(f"  RMSE: ${rmse_xgb_test:,.2f}")
print(f"  R² Score: {r2_xgb_test:.4f}")

print(f"\nXGBoost - Test Set Performance Store 1:")
print(f"""  MAE: ${mean_absolute_error(
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
):,.2f}""")
print(f"""  RMSE: ${np.sqrt(mean_squared_error(
    
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
)):,.2f}""")
print(f"""  R² Score: {r2_score(
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
):.4f}""")

# Feature importance from XGBoost
feature_importance_xgb = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (XGBoost):")
print(feature_importance_xgb.head(10))

fig = go.Figure(data=[
    go.Bar(x=feature_importance_xgb.head(10)['Importance'].values,
           y=feature_importance_xgb.head(10)['Feature'].values,
           orientation='h', marker=dict(color='lightblue'))
])
fig.update_layout(
    title='XGBoost: Top 10 Feature Importances',
    xaxis_title='Importance Score',
    yaxis_title='Feature',
    height=500,
    template='plotly_white'
)
fig.show()


# %% 9.1 XGBOOST MODEL: GRADIENT BOOSTING REGRESSION on Store level
# ============================================================================

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    df_agg_store_train[grp_feature_cols], df_agg_store_train.Weekly_Sales, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: {X_train_split.shape}")
print(f"Validation set shape: {X_val_split.shape}")

# Initialize XGBoost model with optimized hyperparameters
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train with early stopping
xgb_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

print("XGBoost model training complete!")

# Make predictions
y_train_pred_xgb = xgb_model.predict(X_train_split)
y_val_pred_xgb = xgb_model.predict(X_val_split)
y_test_pred_xgb = xgb_model.predict(df_agg_store_test[grp_feature_cols])

train_split_yhat = X_train_split.join(df_agg_store_train["Date"])
train_split_yhat["yhat"] = y_train_pred_xgb
train_split_yhat["Weekly_Sales"] = y_train_split
val_split_yhat = X_val_split.join(df_agg_store_train["Date"])
val_split_yhat["yhat"] = y_val_pred_xgb
val_split_yhat["Weekly_Sales"] = y_val_split
test_split_yhat = df_agg_store_test.copy()
test_split_yhat["yhat"] = y_test_pred_xgb


df_plot_ygrp = pd.concat(
    [
        train_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]], 
        val_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]], 
        test_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]]
    ]
).sort_values("Date").reset_index(drop=True)

df_plot_ygrp[df_plot_ygrp.Store == 1]

plot_time_series(
    x=df_plot_ygrp[df_plot_ygrp.Store == 1].Date.values, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales.values, 
    yhat=df_plot_ygrp[df_plot_ygrp.Store == 1].yhat.values,
    vline_x=train_dates.max(),
    title="XGBoost for all Stores"
)



# Calculate metrics
mae_xgb_train = mean_absolute_error(y_train_split, y_train_pred_xgb)
rmse_xgb_train = np.sqrt(mean_squared_error(y_train_split, y_train_pred_xgb))
r2_xgb_train = r2_score(y_train_split, y_train_pred_xgb)

mae_xgb_val = mean_absolute_error(y_val_split, y_val_pred_xgb)
rmse_xgb_val = np.sqrt(mean_squared_error(y_val_split, y_val_pred_xgb))
r2_xgb_val = r2_score(y_val_split, y_val_pred_xgb)

mae_xgb_test = mean_absolute_error(df_agg_store_test.Weekly_Sales, y_test_pred_xgb)
rmse_xgb_test = np.sqrt(mean_squared_error(df_agg_store_test.Weekly_Sales, y_test_pred_xgb))
r2_xgb_test = r2_score(df_agg_store_test.Weekly_Sales, y_test_pred_xgb)


print(f"\nXGBoost - Training Set Performance:")
print(f"  MAE: ${mae_xgb_train:,.2f}")
print(f"  RMSE: ${rmse_xgb_train:,.2f}")
print(f"  R² Score: {r2_xgb_train:.4f}")

print(f"\nXGBoost - Validation Set Performance:")
print(f"  MAE: ${mae_xgb_val:,.2f}")
print(f"  RMSE: ${rmse_xgb_val:,.2f}")
print(f"  R² Score: {r2_xgb_val:.4f}")

print(f"\nXGBoost - Test Set Performance:")
print(f"  MAE: ${mae_xgb_test:,.2f}")
print(f"  RMSE: ${rmse_xgb_test:,.2f}")
print(f"  R² Score: {r2_xgb_test:.4f}")

print(f"\nXGBoost - Test Set Performance Store 1:")
print(f"""  MAE: ${mean_absolute_error(
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
):,.2f}""")
print(f"""  RMSE: ${np.sqrt(mean_squared_error(
    
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
)):,.2f}""")
print(f"""  R² Score: {r2_score(
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
):.4f}""")


# %% 9.2 XGBOOST MODEL: GRADIENT BOOSTING REGRESSION on Store 1 only
# ============================================================================

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    df_agg_store_train_1[grp_feature_cols], 
    df_agg_store_train_1.Weekly_Sales, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: {X_train_split.shape}")
print(f"Validation set shape: {X_val_split.shape}")

# Initialize XGBoost model with optimized hyperparameters
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train with early stopping
xgb_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

print("XGBoost model training complete!")

# Make predictions
y_train_pred_xgb = xgb_model.predict(X_train_split)
y_val_pred_xgb = xgb_model.predict(X_val_split)
y_test_pred_xgb = xgb_model.predict(df_agg_store_test_1[grp_feature_cols])

train_split_yhat = X_train_split.join(df_agg_store_train_1["Date"])
train_split_yhat["yhat"] = y_train_pred_xgb
train_split_yhat["Weekly_Sales"] = y_train_split
val_split_yhat = X_val_split.join(df_agg_store_train_1["Date"])
val_split_yhat["yhat"] = y_val_pred_xgb
val_split_yhat["Weekly_Sales"] = y_val_split
test_split_yhat = df_agg_store_test_1.copy()
test_split_yhat["yhat"] = y_test_pred_xgb


df_plot_ygrp = pd.concat(
    [
        train_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]], 
        val_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]], 
        test_split_yhat[["Date", "Store", "Weekly_Sales", "yhat"]]
    ]
).sort_values("Date").reset_index(drop=True)

df_plot_ygrp[df_plot_ygrp.Store == 1]

plot_time_series(
    x=df_plot_ygrp[df_plot_ygrp.Store == 1].Date.values, 
    y=df_plot_ygrp[df_plot_ygrp.Store == 1].Weekly_Sales.values, 
    yhat=df_plot_ygrp[df_plot_ygrp.Store == 1].yhat.values,
    vline_x=train_dates.max(),
    title="XGBoost for Store 1 only"
)


# Calculate metrics
mae_xgb_train = mean_absolute_error(y_train_split, y_train_pred_xgb)
rmse_xgb_train = np.sqrt(mean_squared_error(y_train_split, y_train_pred_xgb))
r2_xgb_train = r2_score(y_train_split, y_train_pred_xgb)

mae_xgb_val = mean_absolute_error(y_val_split, y_val_pred_xgb)
rmse_xgb_val = np.sqrt(mean_squared_error(y_val_split, y_val_pred_xgb))
r2_xgb_val = r2_score(y_val_split, y_val_pred_xgb)

mae_xgb_test = mean_absolute_error(df_agg_store_test_1.Weekly_Sales, y_test_pred_xgb)
rmse_xgb_test = np.sqrt(mean_squared_error(df_agg_store_test_1.Weekly_Sales, y_test_pred_xgb))
r2_xgb_test = r2_score(df_agg_store_test_1.Weekly_Sales, y_test_pred_xgb)


print(f"\nXGBoost - Training Set Performance:")
print(f"  MAE: ${mae_xgb_train:,.2f}")
print(f"  RMSE: ${rmse_xgb_train:,.2f}")
print(f"  R² Score: {r2_xgb_train:.4f}")

print(f"\nXGBoost - Validation Set Performance:")
print(f"  MAE: ${mae_xgb_val:,.2f}")
print(f"  RMSE: ${rmse_xgb_val:,.2f}")
print(f"  R² Score: {r2_xgb_val:.4f}")

print(f"\nXGBoost - Test Set Performance:")
print(f"  MAE: ${mae_xgb_test:,.2f}")
print(f"  RMSE: ${rmse_xgb_test:,.2f}")
print(f"  R² Score: {r2_xgb_test:.4f}")

print(f"\nXGBoost - Test Set Performance Store 1:")
print(f"""  MAE: ${mean_absolute_error(
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
):,.2f}""")
print(f"""  RMSE: ${np.sqrt(mean_squared_error(
    
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
)):,.2f}""")
print(f"""  R² Score: {r2_score(
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].Weekly_Sales,
    df_plot_ygrp[(df_plot_ygrp.Store == 1) & (df_plot_ygrp.Date > train_dates.max())].yhat
):.4f}""")


# %%
# 10. MODEL EVALUATION AND COMPARISON
# ============================================================================
# Compare all three models and visualize their performance

print("\n" + "="*70)
print("MODEL COMPARISON: SCALED VS UNSCALED FEATURES")
print("="*70)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': ['LR Train (Scaled)', 'LR Test (Scaled)',
              'Prophet Train', 'Prophet Test',  
              'XGBoost (Train)', 'XGBoost (Validation)', 'XGBoost (Test)'],
    'MAE': [mae_lr, mae_lr_test, mae_pr, mae_pr_test, mae_xgb_train, mae_xgb_val, mae_xgb_test],
    'RMSE': [rmse_lr, rmse_lr_test, rmse_pr, rmse_pr_test, rmse_xgb_train, rmse_xgb_val, rmse_xgb_test],
    'R² Score': [r2_lr, r2_lr_test, r2_pr, r2_pr_test, r2_xgb_train, r2_xgb_val, r2_xgb_test]
})

print("\n--- Model Performance Summary ---")
print(comparison_df.to_string(index=False))

# Visualize model comparison
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('MAE Comparison', 'RMSE Comparison', 'R² Score Comparison')
)

fig.add_trace(
    go.Bar(x=comparison_df['Model'], y=comparison_df['MAE'], name='MAE', marker=dict(color='red')),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'], name='RMSE', marker=dict(color='orange')),
    row=1, col=2
)
fig.add_trace(
    go.Bar(x=comparison_df['Model'], y=comparison_df['R² Score'], name='R² Score', marker=dict(color='green')),
    row=1, col=3
)

fig.update_layout(height=500, template='plotly_white', showlegend=False)
fig.show()

# %%
# 11. PREDICTION VISUALIZATION
# ============================================================================
# Visualize actual vs predicted values to assess model quality

print("\nGenerating prediction visualizations...")

# For XGBoost on validation set
prediction_viz_df = pd.DataFrame({
    'Actual': df_agg_store_test_1.Weekly_Sales,
    'Predicted': y_test_pred_xgb,
    'Residual': df_agg_store_test_1.Weekly_Sales - y_test_pred_xgb
})

# Actual vs Predicted scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=prediction_viz_df['Actual'],
    y=prediction_viz_df['Predicted'],
    mode='markers',
    name='Predictions',
    marker=dict(size=6, color='blue', opacity=0.6)
))

# Perfect prediction line
min_val = min(prediction_viz_df['Actual'].min(), prediction_viz_df['Predicted'].min())
max_val = max(prediction_viz_df['Actual'].max(), prediction_viz_df['Predicted'].max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='XGBoost: Actual vs Predicted Sales (on test)',
    xaxis_title='Actual Sales',
    yaxis_title='Predicted Sales',
    height=500,
    template='plotly_white'
)
fig.show()

# Residuals plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=prediction_viz_df['Actual'],
    y=prediction_viz_df['Residual'],
    mode='markers',
    marker=dict(size=6, color='green', opacity=0.6)
))
fig.add_hline(y=0, line_dash="dash", line_color="red")

fig.update_layout(
    title='XGBoost: Residuals Analysis',
    xaxis_title='Actual Sales',
    yaxis_title='Residual (Actual - Predicted)',
    height=500,
    template='plotly_white'
)
fig.show()
# %%
