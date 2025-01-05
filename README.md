# Tweet Classification Project

This project implements a machine learning pipeline for classifying tweets into different categories (Politics and Sports) using various classification algorithms including Logistic Regression, Random Forest, XGBoost, and Support Vector Machine.

The choice of the dataset is justified by the fact that this is a well known dataset in Kaggle ( a very known competition) and many companies use it for tests in Data Science postiions. So it's a trusted Dataset that will test test our abilities

## Project Structure

```
tweets-classification/
├── data/
│   ├── train_tweets.csv    # Training dataset
│   └── test_tweets.csv     # Test dataset
├── notebook/
│   └── tweets_analysis_v2.ipynb    # Main analysis notebook
```

## Overview

The project is structured as a comprehensive machine learning pipeline that includes:

1. **Data Loading and Initial Analysis**
   - Loading training and test datasets
   - Exploratory data analysis
   - Data schema verification 
   - Distribution analysis of classes
   - Missing value check

2. **Data Preprocessing**
   - Text cleaning using regex
   - Label encoding
   - Text tokenization
   - Stop words removal
   - TF-IDF vectorization

3. **Model Development**
   - Implementation of multiple classifiers:
     - Logistic Regression (maxIter=20)
     - Random Forest (numTrees=10)
     - XGBoost (numRound=100)
     - Support Vector Machine (maxIter=10)

4. **Performance Evaluation**
   - Multiple evaluation metrics:
     - Accuracy
     - F1-Score
     - Precision
     - Recall
   - Confusion matrices for each model
   - Comparative analysis of models

## Key Features

- **PySpark Integration**: Utilizes PySpark for scalable data processing
- **Multi-Model Comparison**: Implements and compares multiple classification algorithms
- **Comprehensive Evaluation**: Uses various metrics to assess model performance
- **Visualization**: Includes confusion matrices and performance comparison plots

## Model Performance

Based on the analysis, the best performing model is:
- **Support Vector Machine**
  - F1-Score: 0.9446
  - Highest overall performance across metrics

## Dependencies

- PySpark
- XGBoost 
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

## Setup and Configuration

The notebook includes Spark configuration settings:
- Driver Memory: 4G
- Executor Memory: 4G
- SQL Shuffle Partitions: 4

## Usage

1. Install required packages:
   ```bash
   pip install sparkxgb
   pip install xgboost
   ```
2. Set up Spark environment with the specified configurations
3. Run the Jupyter notebook tweets_analysis_v2.ipynb

## Results Visualization

The project includes several visualizations:
- Confusion matrices for each model
- Comparative bar plots for different metrics
- Performance metric comparisons across models


## Future Improvements

- **Hyperparameter Optimization**: Apply advanced tuning techniques to improve model performance.
- **Feature Engineering**: Refine feature extraction and selection to enhance data representation.
- **Cross-Validation**: Implement cross-validation to ensure model generalization and reduce overfitting.
- **Advanced Text Preprocessing**: Integrate additional text preprocessing methods for improved data quality.
- **Real-Time Analysis**: Previously implemented real-time analysis for Twitter, now restricted to paid solutions; architecture details are present in this repo (we wanted to do the same idea). https://github.com/Frandiiile/Stock-Market-Real-Time-Analysis 
- **Trend Prediction for Tweets**: Trend prediction is limited due to changes in data access policies, with paid solutions now required.




## Contributers: 
- Ibrahim Mestadi & Aimane Frandile


