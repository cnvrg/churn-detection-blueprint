# SVC [Training]
Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. But generally, they are used in classification problems.
An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. The hyperplane will be generated in an iterative manner by SVM so that the error can be minimized. The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH).
The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is. This makes this specific algorithm rather suitable for our uses, though you can use this for many situations.
### Features
- Upload the file containing the churn data and get a dataframe containing the predicted values of churn and an evaluation metrics file
- User can choose to select from amongst many hyper-parameters to train the model
- User can choose to download the model pickle file and then upload it to batch-predict module or endpoint

# Input Arguments
- `--X_train` path of the training file consisting of independent variables
- `--X_test` path of the test file consisting of independent variables
- `--y_train` path of the training file consisting of the dependent (churn) variable.
- `--y_test` path of the test file consisting of the dependent (churn) variable
- `--kernel` (default = rbf) The main hyperparameter of the SVM is the kernel. It maps the observations into some feature space. Ideally the observations are more easily (linearly) separable after this transformation. There are multiple standard kernels for this transformations, e.g. the linear kernel, the polynomial kernel and the radial kernel. The choice of the kernel and their hyperparameters affect greatly the separability of the classes (in classification) and the performance of the algorithm.
- `--gamma` (default = 1, 0.1, 0.01, 0.001, 0.0001) Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
- `--C` (default = 0.1, 1, 10, 100, 1000) Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.

# Model Artifacts

- `--evaluation.csv` file which contains the evaluation metrics 
    |Algo_Name 	|CV_Accuracy_Score 	|Test_Accuracy 	|Train_Accuracy 	|Test_Precision 	|Train_Precision 	|Test_Recall 	|Train_Recall 	|Test_F1 	|Train_F1 |
    |---|---|---|---|---|---|---|---|---|---|
    |SVC	|0.8010815602836878	|0.7946048272598202	|0.8164300202839757|	0.6388261851015802|	0.7117585848074922|	0.5080789946140036|	0.5213414634146342|	0.7865264173690515|	0.8065027279486526
- `--churn_ouput.csv` file which contains the predicted values, actual values and the independent variables, in a tabluar form

|X-Test-Variables |Y-Test-Variable |Predictions |CV-Predictions | 
|---|---|---|---|
|56 |1 |0 |1 | 
|52.6 |0 |0 |1 |
- `--threshold.csv` file containing the threshold values
- `--my_model.sav` file containing saved model
- `--ROC.png` ROC curve image 

## How to run
```
python3 svc/svc.py --X_train /input/train_test_split/X_train.csv --X_test /input/train_test_split/X_test.csv --y_train /input/train_test_split/y_train.csv --y_test /input/train_test_split/y_test.csv --C 0.1, 1, 10, 100, 1000 --gamma 1, 0.1, 0.01, 0.001, 0.0001 --kernel rbf
```

## About SVC
[Support vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
The main goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH) and it can be done in the following two steps −
- First, SVM will generate hyperplanes iteratively that segregates the classes in best way.
- Then, it will choose the hyperplane that separates the classes correctly.