# Module 20 Report

## Overview of the Analysis

Purpose of the analysis: 

The goal of this analysis is to determine whether the Logistic Regression ML model can more accurately predict healthy loans vs high-risk loans using the original dataset or a dataset resampled to increase the size of the minority class. 

Dataset:

The dataset used to perform the analysis consists of information on 77,536 loans. The data includes columns for loan size, interest rate, borrower's income, debt to income ration, number of accounts, derotatory marks, total debt, and loan status. The category we are attempting to predict with our analysis is loan status. The provided in the remaining columns will be used as features to train the model and inform predictions.

ML Process:
* Prepare the data: import the file, establish the dataframe, evaluate the columns and features.
* Separate the data into features and labels: the labels are what we are attempting to predict -- the status of the loan (healthy, 0; high risk, 1) -- while the features are all of the remaining data used to train and test the model.
* Use train_test_split function to separate the features and labels data into training and testing datasets.
* Import the ML model from the library; in this case, we are importing LogisticRegression from SKLearn.
* Instantiate the model.
* Fit the model using the training data.
* Use model to make a list of predictions using the testing data.
* Evaluate the predictions by calculating and comparing accuracy score, confusion matrix, and classifcation report.

ML Methods Used:
* Primary model: LogisticRegression from SKLearn
* Supporting functions: train_test_split from SKLearn
* Model evaluation: confusion_matrix and classification_report from SKLearn

## Results

* Machine Learning Model - Logistic Regression:
  * Balanced accuracy score: 99%
  * Precision scores:
    * Class 0 (healthy loans): 100%
    * Class 1 (high-risk loans): 84%
  * Recall scores:
    * Class 0 (healthy loans): 99%
    * Class 1 (high-risk loans): 94%

## Summary

The Logistic Regresion model is an outstanding predictor for healthy loans with precision of 100% and recall of 99%, and a very good predictor for unhealthy loans in spite of extreme imbalance between loan types with precision of 84% and recall of 94%. In sum, the Logistic Regression model does a great job at predicting both healthy and high-risk loans based on the features used to train the data.
