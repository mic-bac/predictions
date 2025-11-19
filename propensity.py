"""
================================================================================
CHURN PREDICTION USING PROPENSITY SCORES
================================================================================
Course: Big Data and Machine Learning
Topic: Introduction to Propensity Scores with Python

Learning Objectives:
1. Understand propensity scores in the context of churn prediction
2. Learn complete ML workflow: preparation → modeling → tuning → deployment
3. Build and compare three different ML approaches (linear, tree, neural)
4. Apply hyperparameter tuning and cross-validation
5. Translate ML results into business recommendations

What are Propensity Scores?
---------------------------
In churn prediction, a propensity score is the probability (0-1) that a 
customer will churn based on their characteristics. These scores help:
- Identify high-risk customers for targeted retention campaigns
- Prioritize limited resources efficiently
- Understand which factors drive customer attrition

Dataset: Customer Churn Dataset from Kaggle
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
================================================================================
"""

# %% 1. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb

print("✓ All libraries imported successfully!")
print("\nNotebook Structure:")
print("  1. Import Libraries")
print("  2. Load & Explore Data")
print("  3. Data Preparation")
print("  4. Exploratory Data Analysis")
print("  5. Build Baseline Models (3 types)")
print("  6. Hyperparameter Tuning")
print("  7. Comprehensive Model Comparison")
print("  8. Business Insights & Deployment")
print("\n" + "="*80 + "\n")

# %% 2. LOAD AND EXPLORE DATA
# ============================================================================

print("="*80)
print("STEP 2: DATA LOADING & INITIAL EXPLORATION")
print("="*80)

# Load the dataset
df_train = pd.read_csv('./data/churn/customer_churn_dataset-training-master.csv')
df_test = pd.read_csv("./data/churn/customer_churn_dataset-testing-master.csv")

df = pd.concat([df_train, df_test]).reset_index(drop=True)
df["CustomerID"] = range(len(df))

# %%

print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\n📋 Column Information:")
print(df.info())

print("\n🔍 Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found!")
else:
    print(missing[missing > 0])

df.dropna(inplace=True)

print("\n🎯 Target Variable (Churn) Distribution:")
churn_counts = df['Churn'].value_counts()
churn_rate = df['Churn'].mean() * 100
print(f"  No Churn (0): {churn_counts[0]} ({100-churn_rate:.1f}%)")
print(f"  Churn (1):    {churn_counts[1]} ({churn_rate:.1f}%)")

# %% 3. DATA PREPARATION
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DATA PREPARATION")
print("="*80)

# Remove row with NaN in CustomerID
df.dropna(subset="CustomerID", inplace=True)

# Create working copy
data = df.copy()

# Identify feature types
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from feature lists
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')
if 'Churn' in numerical_cols:
    numerical_cols.remove('Churn')

print(f"\n📊 Feature Types:")
print(f"  Categorical: {len(categorical_cols)} features")
print(f"  Numerical:   {len(numerical_cols)} features")

# Encode categorical variables
print("\n🔄 Encoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
print(f"  ✓ Encoded {len(categorical_cols)} categorical features")

# Prepare features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Remove ID, as it is not a valid predictor
X_train_id = X_train["CustomerID"]
X_train.drop("CustomerID", axis=1, inplace=True)

X_test_id = X_test["CustomerID"]
X_test.drop("CustomerID", axis=1, inplace=True)

print(f"\n📦 Data Split:")
print(f"  Training:   {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% churn)")
print(f"  Testing:    {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% churn)")

# Standardize features (required for LR and NN, not for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  ✓ Features standardized (mean=0, std=1)")

# %% 4. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Visualization 1: Churn Distribution
fig_churn = px.pie(
    df, 
    names='Churn', 
    title='<b>Customer Churn Distribution</b>',
    color='Churn',
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    hole=0.3
)
fig_churn.update_traces(textinfo='percent+label', textfont_size=14)
fig_churn.show()

# Visualization 2: Feature Distributions by Churn
key_features = numerical_cols[:4] if len(numerical_cols) >= 4 else numerical_cols

fig_dist = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'<b>{feat}</b>' for feat in key_features]
)

for i, feat in enumerate(key_features):
    row, col = (i // 2) + 1, (i % 2) + 1
    
    fig_dist.add_trace(
        go.Histogram(x=df[df['Churn']==0][feat], name='No Churn', 
                     marker_color='#2ecc71', opacity=0.7, 
                     legendgroup='g1', showlegend=(i==0)),
        row=row, col=col
    )
    fig_dist.add_trace(
        go.Histogram(x=df[df['Churn']==1][feat], name='Churn', 
                     marker_color='#e74c3c', opacity=0.7,
                     legendgroup='g2', showlegend=(i==0)),
        row=row, col=col
    )

fig_dist.update_layout(
    title_text='<b>Feature Distributions by Churn Status</b>',
    height=600, barmode='overlay'
)
fig_dist.show()

# Visualization 3: Correlation Heatmap
correlation = data.corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=correlation.values,
    x=correlation.columns,
    y=correlation.columns,
    colorscale='RdBu_r',
    zmid=0,
    text=np.round(correlation.values, 2),
    texttemplate='%{text}',
    textfont={"size": 8}
))

fig_corr.update_layout(
    title='<b>Feature Correlation Matrix</b>',
    width=900, height=800
)
fig_corr.show()

# Top correlations with churn
churn_corr = correlation['Churn'].drop('Churn').sort_values(ascending=False)
print("\n🔗 Top 10 Features Correlated with Churn (5 pos, 5 neg):")
for i, (feat, corr) in enumerate(churn_corr.head(5).items(), 1):
    print(f"  {i:2d}. {feat:30s} {corr:+.4f}")
for i, (feat, corr) in enumerate(churn_corr.tail(5).items(), 1):
    print(f"  {i:2d}. {feat:30s} {corr:+.4f}")

# %% 5. BASELINE MODELS (NO TUNING)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: BASELINE MODELS (DEFAULT PARAMETERS)")
print("="*80)
print("Building three model types to establish baselines:\n")

# Dictionary to store all results
models_results = {}

# %% 5.1 Logistic Regression
# ---------------------------------------------------------------------------
print("-" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-" * 80)
print("Linear model that predicts probabilities. Interpretable and fast.")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

models_results['LR_Default'] = {
    'model': lr_model,
    'predictions': lr_pred,
    'probabilities': lr_pred_proba,
    'metrics': {
        'Accuracy': accuracy_score(y_test, lr_pred),
        'Precision': precision_score(y_test, lr_pred),
        'Recall': recall_score(y_test, lr_pred),
        'F1-Score': f1_score(y_test, lr_pred),
        'ROC-AUC': roc_auc_score(y_test, lr_pred_proba)
    }
}

print("\n✓ Performance:")
for metric, value in models_results['LR_Default']['metrics'].items():
    print(f"  {metric:12s}: {value:.4f}")

cm = confusion_matrix(y_test, lr_pred)
print("\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

# %% 5.2 XGBoost
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("MODEL 2: XGBOOST (GRADIENT BOOSTING TREES)")
print("-" * 80)
print("Ensemble of decision trees. Powerful for complex patterns.")

xgb_model = xgb.XGBClassifier(
    random_state=42, eval_metric='logloss',
    max_depth=2, learning_rate=0.01, n_estimators=100
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

models_results['XGB_Default'] = {
    'model': xgb_model,
    'predictions': xgb_pred,
    'probabilities': xgb_pred_proba,
    'metrics': {
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1-Score': f1_score(y_test, xgb_pred),
        'ROC-AUC': roc_auc_score(y_test, xgb_pred_proba)
    }
}

print("\n✓ Performance:")
for metric, value in models_results['XGB_Default']['metrics'].items():
    print(f"  {metric:12s}: {value:.4f}")

cm = confusion_matrix(y_test, xgb_pred)
print("\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

# %% 5.3 Neural Network
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("MODEL 3: NEURAL NETWORK (MULTI-LAYER PERCEPTRON)")
print("-" * 80)
print("Deep learning with 2 hidden layers [32, 16]. Flexible but less interpretable.")

nn_model = MLPClassifier(
    hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
    learning_rate_init=0.2, alpha=0.1, random_state=42, 
    max_iter=100, early_stopping=True
)
nn_model.fit(X_train_scaled, y_train)

nn_pred = nn_model.predict(X_test_scaled)
nn_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]

models_results['NN_Default'] = {
    'model': nn_model,
    'predictions': nn_pred,
    'probabilities': nn_pred_proba,
    'metrics': {
        'Accuracy': accuracy_score(y_test, nn_pred),
        'Precision': precision_score(y_test, nn_pred),
        'Recall': recall_score(y_test, nn_pred),
        'F1-Score': f1_score(y_test, nn_pred),
        'ROC-AUC': roc_auc_score(y_test, nn_pred_proba)
    }
}

print("\n✓ Performance:")
for metric, value in models_results['NN_Default']['metrics'].items():
    print(f"  {metric:12s}: {value:.4f}")

cm = confusion_matrix(y_test, nn_pred)
print("\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

# %% 5.4 Comparison of all 3 model results
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("📊 BASELINE COMPARISON")
print("="*80)

baseline_df = pd.DataFrame({
    name: results['metrics'] 
    for name, results in models_results.items()
})
print("\n", baseline_df.round(4))

# %% 6. HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*80)
print("STEP 6: HYPERPARAMETER TUNING")
print("="*80)
print("""
Hyperparameters control how models learn (not learned from data).
We use cross-validation to find optimal settings systematically.
""")

# %% 6.1 Tune Logistic Regression
# ---------------------------------------------------------------------------
print("-" * 80)
print("Tuning Logistic Regression...")
print("-" * 80)

lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], # regularization to reduce overfitting
    'penalty': ['l1', 'l2'],             # lasso or ridge regression to reduce overfitting
    'solver': ['liblinear', 'saga']      # algorithm to find optimal coefficient
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    lr_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
)
lr_grid.fit(X_train_scaled, y_train)

lr_tuned = lr_grid.best_estimator_
lr_tuned_pred = lr_tuned.predict(X_test_scaled)
lr_tuned_proba = lr_tuned.predict_proba(X_test_scaled)[:, 1]

models_results['LR_Tuned'] = {
    'model': lr_tuned,
    'predictions': lr_tuned_pred,
    'probabilities': lr_tuned_proba,
    'metrics': {
        'Accuracy': accuracy_score(y_test, lr_tuned_pred),
        'Precision': precision_score(y_test, lr_tuned_pred),
        'Recall': recall_score(y_test, lr_tuned_pred),
        'F1-Score': f1_score(y_test, lr_tuned_pred),
        'ROC-AUC': roc_auc_score(y_test, lr_tuned_proba)
    },
    'best_params': lr_grid.best_params_,
    'cv_score': lr_grid.best_score_
}

print(f"Best Parameters: {lr_grid.best_params_}")
print(f"CV ROC-AUC: {lr_grid.best_score_:.4f}")
print(f"Test ROC-AUC: {models_results['LR_Tuned']['metrics']['ROC-AUC']:.4f}")
improvement = models_results['LR_Tuned']['metrics']['ROC-AUC'] - models_results['LR_Default']['metrics']['ROC-AUC']
print(f"Improvement: {improvement:+.4f}")

# %% 6.2 Tune XGBoost
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Tuning XGBoost...")
print("-" * 80)

xgb_param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'min_child_weight': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_param_grid, n_iter=20, cv=5, scoring='roc_auc', 
    n_jobs=-1, verbose=0, random_state=42
)
xgb_random.fit(X_train, y_train)

xgb_tuned = xgb_random.best_estimator_
xgb_tuned_pred = xgb_tuned.predict(X_test)
xgb_tuned_proba = xgb_tuned.predict_proba(X_test)[:, 1]

models_results['XGB_Tuned'] = {
    'model': xgb_tuned,
    'predictions': xgb_tuned_pred,
    'probabilities': xgb_tuned_proba,
    'metrics': {
        'Accuracy': accuracy_score(y_test, xgb_tuned_pred),
        'Precision': precision_score(y_test, xgb_tuned_pred),
        'Recall': recall_score(y_test, xgb_tuned_pred),
        'F1-Score': f1_score(y_test, xgb_tuned_pred),
        'ROC-AUC': roc_auc_score(y_test, xgb_tuned_proba)
    },
    'best_params': xgb_random.best_params_,
    'cv_score': xgb_random.best_score_
}

print(f"Best Parameters: {xgb_random.best_params_}")
print(f"CV ROC-AUC: {xgb_random.best_score_:.4f}")
print(f"Test ROC-AUC: {models_results['XGB_Tuned']['metrics']['ROC-AUC']:.4f}")
improvement = models_results['XGB_Tuned']['metrics']['ROC-AUC'] - models_results['XGB_Default']['metrics']['ROC-AUC']
print(f"Improvement: {improvement:+.4f}")

# %% 6.3 Tune Neural Network
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Tuning Neural Network...")
print("-" * 80)

nn_param_grid = {
    'hidden_layer_sizes': [(32,), (64, 32)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

nn_random = RandomizedSearchCV(
    MLPClassifier(random_state=42, max_iter=200, early_stopping=True),
    nn_param_grid, n_iter=5, cv=3, scoring='roc_auc',
    n_jobs=-1, verbose=0, random_state=42
)
nn_random.fit(X_train_scaled, y_train)

nn_tuned = nn_random.best_estimator_
nn_tuned_pred = nn_tuned.predict(X_test_scaled)
nn_tuned_proba = nn_tuned.predict_proba(X_test_scaled)[:, 1]

models_results['NN_Tuned'] = {
    'model': nn_tuned,
    'predictions': nn_tuned_pred,
    'probabilities': nn_tuned_proba,
    'metrics': {
        'Accuracy': accuracy_score(y_test, nn_tuned_pred),
        'Precision': precision_score(y_test, nn_tuned_pred),
        'Recall': recall_score(y_test, nn_tuned_pred),
        'F1-Score': f1_score(y_test, nn_tuned_pred),
        'ROC-AUC': roc_auc_score(y_test, nn_tuned_proba)
    },
    'best_params': nn_random.best_params_,
    'cv_score': nn_random.best_score_
}

print(f"Best Parameters: {nn_random.best_params_}")
print(f"CV ROC-AUC: {nn_random.best_score_:.4f}")
print(f"Test ROC-AUC: {models_results['NN_Tuned']['metrics']['ROC-AUC']:.4f}")
improvement = models_results['NN_Tuned']['metrics']['ROC-AUC'] - models_results['NN_Default']['metrics']['ROC-AUC']
print(f"Improvement: {improvement:+.4f}")

print("\n✓ Hyperparameter tuning complete!")

# %% 7. COMPREHENSIVE MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("STEP 7: COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    name: results['metrics'] 
    for name, results in models_results.items()
})

print("\n📊 All Models Performance Summary:")
print(comparison_df.round(4))

# Identify best model
best_model_name = comparison_df.loc['ROC-AUC'].idxmax()
best_score = comparison_df.loc['ROC-AUC', best_model_name]
print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")


# %% Visualization: ROC Curves
# ---------------------------------------------------------------------------
model_order = ['LR_Default', 'LR_Tuned', 'XGB_Default', 'XGB_Tuned', 'NN_Default', 'NN_Tuned']
colors = ['#3498db', '#2980b9', '#2ecc71', '#27ae60', '#e74c3c', '#c0392b']

fig_roc = go.Figure()

line_styles = [
    {'dash': 'dash', 'width': 2, 'color': '#3498db'},
    {'dash': 'solid', 'width': 3, 'color': '#2980b9'},
    {'dash': 'dash', 'width': 2, 'color': '#2ecc71'},
    {'dash': 'solid', 'width': 3, 'color': '#27ae60'},
    {'dash': 'dash', 'width': 2, 'color': '#e74c3c'},
    {'dash': 'solid', 'width': 3, 'color': '#c0392b'}
]

for idx, (name, results) in enumerate(models_results.items()):
    fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
    auc = results['metrics']['ROC-AUC']
    
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f"{name.replace('_', ' ')} (AUC={auc:.3f})",
        mode='lines',
        line=line_styles[idx]
    ))

# Reference line
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    name='Random (AUC=0.500)',
    mode='lines',
    line=dict(dash='dot', color='gray', width=1)
))

fig_roc.update_layout(
    title='<b>ROC Curves: All Models (Dashed=Default, Solid=Tuned)</b>',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=600,
    width=900
)
fig_roc.show()

# %% Visualization: Propensity Score Distributions
# ---------------------------------------------------------------------------
fig_prop = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"<b>{name.replace('_', ' ')}</b>" for name in model_order],
    vertical_spacing=0.12
)

for idx, model_name in enumerate(model_order):
    row = (idx // 3) + 1
    col = (idx % 3) + 1
    
    proba = models_results[model_name]['probabilities']
    
    # No Churn
    fig_prop.add_trace(
        go.Histogram(
            x=proba[y_test == 0],
            name='No Churn',
            marker_color='#2ecc71',
            opacity=0.7,
            legendgroup='g1',
            showlegend=(idx == 0),
            nbinsx=30
        ),
        row=row, col=col
    )
    
    # Churn
    fig_prop.add_trace(
        go.Histogram(
            x=proba[y_test == 1],
            name='Churn',
            marker_color='#e74c3c',
            opacity=0.7,
            legendgroup='g2',
            showlegend=(idx == 0),
            nbinsx=30
        ),
        row=row, col=col
    )

fig_prop.update_xaxes(title_text="Propensity Score", range=[0, 1])
fig_prop.update_yaxes(title_text="Count")
fig_prop.update_layout(
    title_text='<b>Propensity Score Distributions by Model</b>',
    barmode='overlay',
    height=700,
    width=1400
)
fig_prop.show()


# %% Feature Importance (Top Models)
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Feature Importance Analysis")
print("-" * 80)

# Logistic Regression Coefficients
lr_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': np.abs(models_results['LR_Tuned']['model'].coef_[0])
}).sort_values('Coefficient', ascending=False).head(10)

# XGBoost Feature Importance
xgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': models_results['XGB_Tuned']['model'].feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print("\n📊 Top 10 Features - Logistic Regression:")
for i, row in lr_importance.iterrows():
    print(f"  {row['Feature']:30s} {row['Coefficient']:.4f}")

print("\n📊 Top 10 Features - XGBoost:")
for i, row in xgb_importance.iterrows():
    print(f"  {row['Feature']:30s} {row['Importance']:.4f}")

# Visualize side by side
fig_importance = make_subplots(
    rows=1, cols=2,
    subplot_titles=('<b>Logistic Regression</b>', '<b>XGBoost</b>')
)

fig_importance.add_trace(
    go.Bar(
        x=lr_importance['Coefficient'],
        y=lr_importance['Feature'],
        orientation='h',
        marker_color='#3498db',
        name='LR Coefficients'
    ),
    row=1, col=1
)

fig_importance.add_trace(
    go.Bar(
        x=xgb_importance['Importance'],
        y=xgb_importance['Feature'],
        orientation='h',
        marker_color='#2ecc71',
        name='XGB Importance'
    ),
    row=1, col=2
)

fig_importance.update_xaxes(title_text="Coefficient Magnitude", row=1, col=1)
fig_importance.update_xaxes(title_text="Importance Score", row=1, col=2)
fig_importance.update_yaxes(autorange="reversed")

fig_importance.update_layout(
    title_text='<b>Top 10 Feature Importance Comparison</b>',
    height=500,
    width=1200,
    showlegend=False
)
fig_importance.show()

# %% Detailed Classification Reports for Best Models
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS - TUNED MODELS")
print("="*80)

model_names_for_report = ['LR_Tuned', 'XGB_Tuned', 'NN_Tuned']

for model_name in model_names_for_report:
    predictions = models_results[model_name]['predictions']
    
    print(f"\n{'-'*80}")
    print(f"📋 {model_name.replace('_', ' ')}")
    print(f"{'-'*80}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              No Churn  Churn")
    print(f"Actual  No Churn  {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"        Churn    {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN):  {tn} - Correctly predicted no churn")
    print(f"  False Positives (FP): {fp} - Incorrectly predicted churn")
    print(f"  False Negatives (FN): {fn} - Incorrectly predicted no churn (MISSED churners)")
    print(f"  True Positives (TP):  {tp} - Correctly predicted churn")
    print(f"  Specificity: {specificity:.4f} (% of non-churners correctly identified)")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                              target_names=['No Churn', 'Churn'],
                              digits=4))
    
# %% 8. BUSINESS INSIGHTS & DEPLOYMENT RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 8: BUSINESS INSIGHTS & DEPLOYMENT")
print("="*80)

# Use best model for business recommendations
best_model_results = models_results[best_model_name]
best_propensity = best_model_results['probabilities']

print(f"\n🎯 Selected Model for Deployment: {best_model_name.replace('_', ' ')}")
print(f"   Performance Metrics:")
for metric, value in best_model_results['metrics'].items():
    print(f"     {metric:12s}: {value:.4f}")

# %% Customer Risk Segmentation
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Customer Risk Segmentation")
print("-" * 80)

# Define risk thresholds
threshold_high = 0.7
threshold_medium = 0.4

# Create risk segments
risk_segments = pd.DataFrame({
    'Customer_ID': range(len(y_test)),
    'Propensity_Score': best_propensity,
    'Actual_Churn': y_test.values
})

risk_segments['Risk_Segment'] = pd.cut(
    risk_segments['Propensity_Score'],
    bins=[0, threshold_medium, threshold_high, 1],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

print("\n📊 Customer Distribution by Risk Segment:")
segment_stats = risk_segments.groupby('Risk_Segment').agg({
    'Customer_ID': 'count',
    'Actual_Churn': 'mean'
}).rename(columns={'Customer_ID': 'Count', 'Actual_Churn': 'Actual_Churn_Rate'})
segment_stats['Actual_Churn_Rate'] = segment_stats['Actual_Churn_Rate'] * 100

for segment in ['Low Risk', 'Medium Risk', 'High Risk']:
    if segment in segment_stats.index:
        count = segment_stats.loc[segment, 'Count']
        rate = segment_stats.loc[segment, 'Actual_Churn_Rate']
        pct = (count / len(risk_segments)) * 100
        print(f"  {segment:12s}: {count:5.0f} customers ({pct:5.1f}%) - Actual churn: {rate:5.1f}%")

# Visualize risk segments
fig_segments = px.histogram(
    risk_segments,
    x='Risk_Segment',
    color='Actual_Churn',
    barmode='group',
    title='<b>Customer Distribution by Risk Segment</b>',
    labels={'Actual_Churn': 'Churned', 'Risk_Segment': 'Risk Segment'},
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    category_orders={'Risk_Segment': ['Low Risk', 'Medium Risk', 'High Risk']}
)
fig_segments.update_layout(height=500)
fig_segments.show()

# %% Cross-Validation Analysis
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Cross-Validation Robustness Check")
print("-" * 80)

print("\nPerforming 5-fold cross-validation on tuned models...")

# Get appropriate data for each model type
cv_data = {
    'LR_Tuned': (X_train_scaled, y_train, models_results['LR_Tuned']['model']),
    'XGB_Tuned': (X_train, y_train, models_results['XGB_Tuned']['model']),
    'NN_Tuned': (X_train_scaled, y_train, models_results['NN_Tuned']['model'])
}

cv_results = {}
for name, (X_data, y_data, model) in cv_data.items():
    scores = cross_val_score(model, X_data, y_data, cv=5, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"\n{name.replace('_', ' ')}:")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Range:  [{scores.min():.4f}, {scores.max():.4f}]")

# Visualize CV results
cv_df = pd.DataFrame(cv_results)
cv_melted = cv_df.melt(var_name='Model', value_name='ROC-AUC')

fig_cv = px.box(
    cv_melted,
    x='Model',
    y='ROC-AUC',
    color='Model',
    title='<b>Cross-Validation Stability (5 Folds)</b>',
    points='all',
    color_discrete_map={
        'LR_Tuned': '#3498db',
        'XGB_Tuned': '#2ecc71',
        'NN_Tuned': '#e74c3c'
    }
)
fig_cv.update_xaxes(title_text='Model')
fig_cv.update_yaxes(title_text='ROC-AUC Score', range=[0.5, 1.0])
fig_cv.update_layout(height=500, showlegend=False)
fig_cv.show()

# %% Deployment Recommendations
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("DEPLOYMENT RECOMMENDATIONS")
print("="*80)

print("""
📋 IMPLEMENTATION ROADMAP

1. MODEL DEPLOYMENT
   ✓ Champion Model: {champion}
   ✓ Test Set Performance: {performance:.4f} ROC-AUC
   ✓ Cross-Validation Stability: {cv_mean:.4f} ± {cv_std:.4f}
   
2. SCORING PIPELINE
   • Frequency: Daily batch scoring
   • Input: Customer database (latest features)
   • Output: Propensity scores + risk segments
   • Storage: Update customer table with scores & segments
   
3. BUSINESS ACTIONS BY RISK SEGMENT
   
   🔴 HIGH RISK (Score > {high_thresh})
      → Immediate outreach by retention team
      → Premium retention offers (discounts, upgrades)
      → Executive escalation for high-value customers
      → Target: Reduce churn by 30-40%
      
   🟡 MEDIUM RISK (Score {med_thresh}-{high_thresh})
      → Automated engagement campaigns (email, SMS)
      → Customer satisfaction surveys
      → Product usage tips and tutorials
      → Target: Prevent escalation to high risk
      
   🟢 LOW RISK (Score < {med_thresh})
      → Standard communications
      → Upsell opportunities
      → Loyalty program engagement
      → Target: Maintain satisfaction
      
4. MONITORING & MAINTENANCE
   
   📊 Weekly Monitoring:
      • Track model performance (precision, recall, ROC-AUC)
      • Monitor feature distributions for data drift
      • A/B test retention strategies
      
   🔄 Monthly Review:
      • Analyze false positives/negatives
      • Gather feedback from retention team
      • Calculate campaign ROI
      
   🔧 Quarterly Retraining:
      • Retrain with last 12 months data
      • Re-tune hyperparameters
      • Update if performance drops >5%
      
5. SUCCESS METRICS
   
   Model Metrics:
   • Maintain ROC-AUC > {target_auc:.2f}
   • Precision > 70% (minimize false alarms)
   • Recall > 60% (catch most churners)
   
   Business Metrics:
   • Churn rate reduction: Target 15-20%
   • Retention campaign ROI: Target 3:1
   • Customer lifetime value increase
   • Retention team efficiency improvement

6. ETHICAL CONSIDERATIONS
   • Ensure fair treatment across customer segments
   • Avoid discrimination based on protected attributes
   • Transparent communication with customers
   • Regular bias audits

""".format(
    champion=best_model_name.replace('_', ' '),
    performance=best_model_results['metrics']['ROC-AUC'],
    cv_mean=cv_results[best_model_name].mean() if best_model_name in cv_results else best_model_results['metrics']['ROC-AUC'],
    cv_std=cv_results[best_model_name].std() if best_model_name in cv_results else 0,
    high_thresh=threshold_high,
    med_thresh=threshold_medium,
    target_auc=best_model_results['metrics']['ROC-AUC'] * 0.95
))

# ---------------------------------------------------------------------------
# Final Summary Table
# ---------------------------------------------------------------------------
print("="*80)
print("📊 FINAL MODEL SUMMARY")
print("="*80)

summary_table = []
for name, results in models_results.items():
    model_type = name.split('_')[0]
    version = name.split('_')[1]
    summary_table.append({
        'Model': model_type,
        'Version': version,
        'ROC-AUC': results['metrics']['ROC-AUC'],
        'Precision': results['metrics']['Precision'],
        'Recall': results['metrics']['Recall'],
        'F1-Score': results['metrics']['F1-Score']
    })

summary_df = pd.DataFrame(summary_table)
print("\n", summary_df.to_string(index=False))

# Calculate improvements
print("\n" + "-"*80)
print("💡 KEY IMPROVEMENTS FROM TUNING:")
print("-"*80)
for model_type in ['LR', 'XGB', 'NN']:
    default_auc = models_results[f'{model_type}_Default']['metrics']['ROC-AUC']
    tuned_auc = models_results[f'{model_type}_Tuned']['metrics']['ROC-AUC']
    improvement = tuned_auc - default_auc
    pct_improvement = (improvement / default_auc) * 100
    
    model_name = {'LR': 'Logistic Regression', 'XGB': 'XGBoost', 'NN': 'Neural Network'}[model_type]
    print(f"{model_name:20s}: {default_auc:.4f} → {tuned_auc:.4f} "
          f"(+{improvement:.4f}, +{pct_improvement:.1f}%)")
