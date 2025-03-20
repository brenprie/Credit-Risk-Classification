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
* To create resampled, balanced dataset, use RandomOverSampler module from the imbalanced-learn library.

ML Methods Used:
* Primary model: LogisticRegression from SKLearn
* Supporting functions: train_test_split from SKLearn
* Model evaluation: balanced accuracy score; confusion_matrix and classification_report from SKLearn
* Resampling module: RandomOverSampler from imblearn.over_sampling

## Results

* ML Model 1 - Logistic Regression, Original Data:
  * Balanced Accuracy Score: 99% → Indicates that the model is extremely balanced in its predictions across both classes.
  * Confusion Matrix:
    * [[18655   110]  → Healthy loans (0): 18,655 correctly predicted, 110 misclassified
    * [    36   583]] → High-risk loans (1): 583 correctly predicted, 36 misclassified
  * Precision scores:
    * Class 0 (healthy loans): 100% → When the model predicts a loan as healthy, it is correct 100% of the time.
    * Class 1 (high-risk loans): 84% → When the model predicts a loan as high-risk, it is correct 84% of the time.
  * Recall scores:
    * Class 0 (healthy loans): 99% → The model captures almost all actual healthy loans.
    * Class 1 (high-risk loans): 94% → The model captures 94% of actual high-risk loans.
   
* ML Model 2 - Logistic Regression, Resampled Data:
  * Balanced Accuracy Score: 99% → The model is again extremely balanced in its predictions across both classes.
  * Confusion Matrix:  
    * [[18646   119]  → Healthy loans (0): 18,646 correctly predicted, 119 misclassified
    * [    4   615]] → High-risk loans (1): 615 correctly predicted, only 4 misclassified
    * The resampled model almost perfectly predicts high-risk loans, with only 4 false negatives, a significant improvement in recall.
    * Misclassification of healthy loans as high-risk increases from 110 to 119, a reasonable trade-off for the gain in recall for high-risk loans.
  * Precision scores:
    * Class 0 (healthy loans): 100% (unchanged)
    * Class 1 (high-risk loans): 84% (unchanged)
  * Recall scores:
    * Class 0 (healthy loans): 99% → The model again captures almost all actual healthy loans.
    * Class 1 (high-risk loans): 99% → The model now captures almost all actual high-risk loans.

## Summary

The Logistic Regresion model is an outstanding predictor for healthy loans, both in the orginal data and in the resampled, balanced data. 

Both models achieve an extremely high 99% balanced accuracy, indicating excellent overall performance.

Precision scores are 100% for healthy loans and 84% for unhealthy loans in both cases, which means both the orginal data and the resampled data provide similar precision (when the model predicts a loan as healthy/high risk, it is correct 100%/84% of the time). Recall for healthy loans is also unchanged at 99% across both samples, but it increases from 94% to 99% for unhealthy loans when switching from the original to resampled data, which means that under the resampled data the model detects almost all actual high-risk loans. The resampled data thus provides some predictive improvement over the original data at no cost in Precision or Recall where healthy loans are concerned. 

In sum, using the resampled data, the model is notably better at identifying high-risk loans, reducing the risk of false negatives, which is crucial in financial risk assessment.

Additional insights: while the improvement in Recall for high-risk loans is impressive using the resampled data, the Precision remains unchanged at 84%, which means that the resampled model continues to have false positives -- healthy loans misclassified as high risk, which can lead to the denial of a good/profitable loan. Further analysis might explore adjusting the classification/decision threshold.

