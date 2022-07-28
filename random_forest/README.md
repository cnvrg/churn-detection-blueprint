# Random Forest [Training]
Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance.
### Features
- Upload the file containing the churn data and get a dataframe containing the predicted values of churn and an evaluation metrics file
- User can choose to select from amongst many hyper-parameters to train the model
- User can choose to download the model pickle file and then upload it to batch-predict module or endpoint

# Input Arguments
- `--X_train` path of the training file consisting of independent variables
- `--X_test` path of the test file consisting of independent variables
- `--y_train` path of the training file consisting of the dependent (churn) variable.
- `--y_test` path of the test file consisting of the dependent (churn) variable
- `--max_features` (default = auto, sqrt) These are the maximum number of features Random Forest is allowed to try in individual tree.
- `--n_estimators` (int, default = 10) This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower. You should choose as high value as your processor can handle because this makes your predictions stronger and more stable.
- `--min_sample_leaf` (default = 2, 5) If you have built a decision tree before, you can appreciate the importance of minimum sample leaf size. Leaf is the end node of a decision tree. A smaller leaf makes the model more prone to capturing noise in train data. Generally I prefer a minimum leaf size of more than 50. However, you should try multiple leaf sizes to find the most optimum for your use case.
- `--min_samples_split` (default = 1, 2)min number of data points placed in a node before the node is split
- `--bootstrap ` (default = True, False)  method for sampling data points (with or without replacement)

# Model Artifacts

- `--evaluation.csv` file which contains the evaluation metrics 
    |Algo_Name 	|CV_Accuracy_Score 	|Test_Accuracy 	|Train_Accuracy 	|Test_Precision 	|Train_Precision 	|Test_Recall 	|Train_Recall 	|Test_F1 	|Train_F1 |
    |---|---|---|---|---|---|---|---|---|---|
    |Random_Forest	|0.791427304964539	|0.7856128726928537		|0.9983772819472616	|0.6150442477876106	|0.9961948249619482	|0.4991023339317774	|0.9977134146341463	|0.9977134146341463	|0.9977134146341463
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
python3 random_forest/RF.py --X_train /input/train_test_split/X_train.csv --X_test /input/train_test_split/X_test.csv --y_train /input/train_test_split/y_train.csv --y_test /input/train_test_split/y_test.csv --n_estimators 10 --max_features auto, sqrt --max_depth 2,4 --min_samples_split 2,5 --min_samples_leaf 1,2 --bootstrap True, False 
```

## About Random Forest
[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
A random forest classifier
In random forests (see RandomForestClassifier and RandomForestRegressor classes), each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set.
Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features. 
The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice the variance reduction is often significant hence yielding an overall better model.