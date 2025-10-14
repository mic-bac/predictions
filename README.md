# Practical Machine Learning Predictions

This repository contains educational examples of common prediction tasks in machine learning and data science. It's designed to help students and beginners understand key concepts through practical, real-world examples.

## 🎯 Learning Objectives

- Understand different types of prediction problems in machine learning
- Learn how to work with real-world datasets
- Gain practical experience with popular ML libraries and frameworks
- Master visualization techniques for data analysis
- Apply best practices in machine learning workflows

## 🗂️ Project Structure

The repository includes three main prediction examples:

### 1. Customer Churn Prediction (Propensity Analysis)
`propensity.py`
- Probability-based approach to predict customer churn
- Uses logistic regression, neural networks, and other ML models
- Focuses on model comparison and evaluation
- Includes feature importance analysis
- Demonstrates complete ML workflow from data prep to deployment

### 2. Customer Churn Analysis (Survival Analysis)
`survival.py`
- Time-based approach to analyze customer churn
- Uses survival analysis techniques (Kaplan-Meier, Cox models)
- Handles censored data (active customers)
- Provides time-to-event predictions
- Visualizes survival curves and hazard rates

### 3. Sales Forecasting (Time Series)
`timeseries.py`
- Time series forecasting for retail sales
- Uses Facebook Prophet and XGBoost
- Includes seasonal decomposition
- Features advanced visualization techniques
- Demonstrates feature engineering for time series

## 🛠️ Prerequisites

- Python 3.10 or later
- Basic understanding of Python programming
- Familiarity with data analysis concepts
- Basic statistics knowledge

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/mic-bac/predictions.git
cd predictions
```

2. Create a conda environment using the provided configuration:
```bash
conda env create -f conda_env.yaml
```

3. Activate the environment:
```bash
conda activate predictions
```

## 📊 Datasets

The project uses two main datasets:

### Customer Churn Dataset
Located in `data/churn/`:
- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Source: [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

### Walmart Sales Dataset
Located in `data/sales/`:
- `train.csv`: Historical sales data
- `test.csv`: Test dataset
- `stores.csv`: Store metadata
- `features.csv`: Additional features (temperature, fuel price, etc.)

Source: [Kaggle Walmart Sales Dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

## 📚 Getting Started

Each Python file is self-contained and includes detailed comments explaining the concepts and implementation:

1. For churn prediction using propensity scores:
```python
python propensity.py
```

2. For survival analysis approach to churn:
```python
python survival.py
```

3. For time series sales forecasting:
```python
python timeseries.py
```

## 📋 Dependencies

Main libraries used:
- pandas & numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- scikit-survival: Survival analysis
- prophet: Time series forecasting
- xgboost: Gradient boosting
- plotly: Interactive visualizations
- statsmodels: Statistical models and tests

## 🎓 Learning Path

1. Start with `propensity.py` to learn basic ML workflow and classification
2. Move to `survival.py` to understand time-based analysis
3. Finally, explore `timeseries.py` for forecasting techniques

Each file includes:
- Detailed comments explaining concepts
- Step-by-step implementation
- Visualization of results
- Model evaluation metrics
- Business insights interpretation

## 🤝 Contributing

Contributions to improve the educational content or add new examples are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


