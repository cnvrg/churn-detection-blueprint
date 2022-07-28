# Logistic Regression [Training]
Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome; something that can take two values such as true/false, yes/no, and so on. Multinomial logistic regression can model scenarios where there are more than two possible discrete outcomes. 
Logistic regression, despite its name, is a classification model rather than regression model. Logistic regression essentially uses a logistic function to model a binary variable, and its range is bounded between 0 and 1. It uses gradient descent among other techniques to decrease its cost function. Log-Loss is generally used as the cost function for logistic regression, though others available as well.
### Features
- Upload the file containing the churn data and get a dataframe containing the predicted values of churn and an evaluation metrics file
- User can choose to select from amongst many hyper-parameters to train the model
- User can choose to download the model pickle file and then upload it to batch-predict module or endpoint

# Input Arguments
- `--X_train` path of the training file consisting of independent variables
- `--X_test` path of the test file consisting of independent variables
- `--y_train` path of the training file consisting of the dependent (churn) variable.
- `--y_test` path of the test file consisting of the dependent (churn) variable
- `--penalty` (default = elasticnet)-specifies the norm of the regularization penalty
- `--l1_ratio` (float, default = 0.5) -he Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
- `--solver` Algorithm to use in the optimization problem. Default is ‘saga’.For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones; ‘liblinear’ is limited to one-versus-rest schemes. Warning The choice of the algorithm depends on the penalty chosen: Supported penalties by solver:
‘newton-cg’ - [‘l2’, ‘none’]
‘lbfgs’ - [‘l2’, ‘none’]
‘liblinear’ - [‘l1’, ‘l2’]
‘sag’ - [‘l2’, ‘none’]
‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, ‘none’]
- `--max_iter` Maximum number of iterations taken for the solvers to converge.
- `--inv_reg_strength`  Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
- `--threshold` (default = 0.5) percentage value beyond which the customer is considered churned and below which, the customer is considered "not churned"

# Model Artifacts
- `--churn_ouput.csv` file which contains the predicted values, actual values and the independent variables, in a tabluar form

|X-Test-Variables |Y-Test-Variable |Predictions |CV-Predictions | 
|---|---|---|---|
|56 |1 |0 |1 | 
|52.6 |0 |0 |1 |
- `--evaluation.csv` file which contains the evaluation metrics 
    |Algo_Name 	|CV_Accuracy_Score 	|Test_Accuracy 	|Train_Accuracy 	|Test_Precision 	|Train_Precision 	|Test_Recall 	|Train_Recall 	|Test_F1 	|Train_F1 |
    |---|---|---|---|---|---|---|---|---|---|
    |Logistic_Regression	|0.8040643133462282	|0.7813535257927118	|0.791683569979716	|0.5716440422322775	|0.5976696367374914	|0.6804308797127468	|0.6646341463414634	|0.7869970578421411	|0.7950492436035441
- `--threshold.csv` file containing the threshold values
- `--important_metrics.csv` file containing important metrics

| variable          | coefficient          | Median  | Yes-No     | String     |
|-------------------|----------------------|---------|------------|------------|
| tenure            | -0.9572776622928866  |         | Partner    | customerID |
| InternetService   | 0.5359179164434023   | tenure  | Dependents | gender     |
| Contract-Two year | -0.29955476433809725 |         |            |            |
- `--my_model.sav` file containing saved model
- `--ROC.png` ROC curve image 
## How to run
```
python3 logistic_regression/LR.py --X_train /input/train_test_split/X_train.csv --X_test /input/train_test_split/X_test.csv --y_train /input/train_test_split/y_train.csv --y_test /input/train_test_split/y_test.csv --penalty elasticnet --l1_ratio 0.5 --solver saga --max_iter 100 --inv_reg_strength 0.8 --threshold 0.4 
```

## About Logistic Regression
[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
Logistic Regression (aka logit, MaxEnt) classifier.
In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)
This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default. It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied).
The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation, or no regularization. The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. The Elastic-Net regularization is only supported by the ‘saga’ solver.