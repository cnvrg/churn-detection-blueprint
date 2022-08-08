# Naive Bayes Classifiers
Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.
### Features
- Upload the file containing the churn data and get a dataframe containing the predicted values of churn and an evaluation metrics file
- User can choose to select from amongst many hyper-parameters to train the model
- User can choose to download the model pickle file and then upload it to batch-predict module or endpoint

# Input Arguments
- `--X_train` path of the training file consisting of independent variables
- `--X_test` path of the test file consisting of independent variables
- `--y_train` path of the training file consisting of the dependent (churn) variable.
- `--y_test` path of the test file consisting of the dependent (churn) variable
- `--var_smoothing` (default = np.logspace(0,-9, num=100)) var_smoothing is a stability calculation to widen (or smooth) the curve and therefore account for more samples that are further away from the distribution mean. In this case, np.logspace returns numbers spaced evenly on a log scale, starts from 0, ends at -9, and generates 100 samples.


# Model Artifacts

- `--evaluation.csv` file which contains the evaluation metrics 
    |Algo_Name 	|CV_Accuracy_Score 	|Test_Accuracy 	|Train_Accuracy 	|Test_Precision 	|Train_Precision 	|Test_Recall 	|Train_Recall 	|Test_F1 	|Train_F1 |
    |---|---|---|---|---|---|---|---|---|---|
    |Naive_Bayes|	0.7073726627981947|	0.7084713677236157|	0.7042596348884381|	0.47070506454816285|	0.4685073339085418|	0.8509874326750448|	0.8277439024390244|	0.7257703786695328|	0.7213620430128561
- `--churn_ouput.csv` file which contains the predicted values, actual values and the independent variables, in a tabluar form

|X-Test-Variables |Y-Test-Variable |Predictions |CV-Predictions | 
|---|---|---|---|
|56 |1 |0 |1 | 
|52.6 |0 |0 |1 |
- `--threshold.csv` file containing the threshold values
- `--important_metrics.csv` file containing important metrics

| variable          | coefficient          |
|-------------------|----------------------|
| tenure            | 0.0651206814955041  |
| DeviceProtection-No   | -0.00444865120681488   |
| InternetService | -0.003502129673450005|
- `--my_model.sav` file containing saved model
- `--ROC.png` ROC curve image 
## How to run
```
python3 naive_bayes/nb.py --X_train /input/train_test_split/X_train.csv --X_test /input/train_test_split/X_test.csv --y_train /input/train_test_split/y_train.csv --y_test /input/train_test_split/y_test.csv --var_smoothing  np.logspace(0,-9, num=100)
```

## About Naive Bayes
[Naive Bayes]https://scikit-learn.org/stable/modules/naive_bayes.html)
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.
In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. They require a small amount of training data to estimate the necessary parameters. 