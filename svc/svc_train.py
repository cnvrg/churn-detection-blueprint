import joblib
from cnvrgv2 import Cnvrg, LineChart
import matplotlib.pyplot as plt
from numpy import mean
from cnvrg.charts import MatrixHeatmap
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict
from sklearn.feature_selection import RFECV
from cnvrg import Experiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.svm import SVC
import argparse
import numpy as np
import pandas as pd
import warnings
import random
from sklearn.exceptions import DataConversionWarning
import os
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "-f",
    "--X_train",
    action="store",
    dest="X_train",
    default="/input/train_test_split/X_train.csv",
    required=True,
    help="""X_train""",
)
parser.add_argument(
    "--X_test",
    action="store",
    dest="X_test",
    default="/input/train_test_split/X_test.csv",
    required=True,
    help="""X_test""",
)
parser.add_argument(
    "--y_train",
    action="store",
    dest="y_train",
    default="/input/train_test_split/y_train.csv",
    required=True,
    help="""y_train""",
)
parser.add_argument(
    "--y_test",
    action="store",
    dest="y_test",
    default="/input/train_test_split/y_test.csv",
    required=True,
    help="""y_test""",
)
parser.add_argument(
    "--gamma",
    action="store",
    dest="gamma",
    default="elasticnet",
    required=True,
    help="""regularization gamma""",
)
parser.add_argument(
    "--kernel",
    action="store",
    dest="kernel",
    default="0.5",
    required=True,
    help="""kernel""",
)
parser.add_argument(
    "--inv_reg_strength",
    action="store",
    dest="inv_reg_strength",
    default="0.8",
    required=True,
    help="""degree of regularization strength""",
)

parser.add_argument(
    "--threshold",
    action="store",
    dest="threshold",
    default="0.5",
    required=True,
    help="""% beyond which the value will be considered to be churned""",
)
parser.add_argument(
    "--id_column",
    action="store",
    dest="id_column",
    default=" ",
    required=True,
    help="""id column""",
)
############################################## Arguments #########################################
args = parser.parse_args()
X_train = args.X_train
X_test = args.X_test
y_train = args.y_train
y_test = args.y_test
gamma = args.gamma
kernel = args.kernel
inv_reg_strength = float(args.inv_reg_strength)
threshold = float(args.threshold)
id_column = args.id_column.split(',')


################## Defining Modelling Function and evaluation metrics function ###################

def modeling(alg, alg_name, gamma, C, kernel, threshold):
    model = alg(gamma=gamma, C=C, kernel=kernel, probability=True)
    model.fit(X_train, y_train.values.ravel())
    X = X_train.append(X_test, ignore_index=True)
    y = y_train.append(y_test, ignore_index=True)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    y_cv_pred = cross_val_predict(model, X, y.values.ravel(), cv=cv)
    scores = cross_val_score(model, X, y.values.ravel(),
                             scoring='accuracy', cv=cv, n_jobs=-1)

    y_pred = np.where(model.predict_proba(X_test)[
                      :, 1] > threshold, 1, 0)  # model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_train = np.where(model.predict_proba(
        X_train)[:, 1] > threshold, 1, 0)  # model.predict(X_train)

    # Performance evaluation
    def print_scores(alg, y_true, y_pred):

        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ", acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ", pre_score)
        rec_score = recall_score(y_true, y_pred)
        print("recall: ", rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ", f_score)

        return acc_score, pre_score, rec_score, f_score

    acc_score, pre_score, rec_score, f_score = print_scores(
        alg, y_test, y_pred)
    acc_score_train, pre_score_train, rec_score_train, f_score_train = print_scores(
        alg, y_train, y_pred_train)
    return model, y_pred_proba, y_pred, y_test, acc_score, pre_score, rec_score, f_score, y_cv_pred, scores, y_pred_train, acc_score_train, pre_score_train, rec_score_train, f_score_train

####################### getting the training and test data and cleaning it ######################
X_train = pd.read_csv(X_train)
X_test = pd.read_csv(X_test)
y_train = pd.read_csv(y_train)
y_test = pd.read_csv(y_test)
X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

####################################### removing ID column ######################################
id_column_list = []
if id_column != ['None']:
    for i in range(len(id_column)):
        id_column_list.append(X_test[id_column[i]])
        X_test.drop([id_column[i]], axis=1, inplace=True)
    for i in range(len(id_column)):
        X_train.drop([id_column[i]], axis=1, inplace=True)

################################# running the modelling function ################################

svc_model, y_pred_proba, y_pred, y_test, acc_score, pre_score, rec_score, f_score, y_cv_pred, scores, y_pred_train, acc_score_train, pre_score_train, rec_score_train, f_score_train = modeling(
    SVC, 'SVC', gamma, inv_reg_strength, kernel, threshold)

filename = cnvrg_workdir+'/my_model.sav'
joblib.dump(svc_model, filename)
e = Experiment()
if kernel == 'linear':
    importance = svc_model.coef_
    df_1 = pd.DataFrame({'variable':X_train.columns,'coefficient':importance[0]})
    var_imp_svc = df_1.sort_values(by='coefficient', key=abs, ascending=False).reset_index(drop=True)

    for i in range(min(var_imp_svc.shape[0],5)):
        e.log_param('metric-'+str(i)+var_imp_svc['variable'][i],var_imp_svc['coefficient'][i])

############### converting arrays/lists to dataframes and printing the dataframes ###############

y_pred = pd.DataFrame(y_pred, columns=['Predictions'])
y_pred_proba_df = pd.DataFrame([round(elem, 4) for elem in y_pred_proba], columns=['Prediction Probability'])
y_cv_pred = pd.DataFrame(y_cv_pred, columns=['CV_Predictions'])

X_test.reset_index(drop=True, inplace=True)

if id_column != ['None']:
    for i in range(len(id_column)):
        X_test[id_column[i]] = id_column_list[i]
        
y_test.reset_index(drop=True, inplace=True)
result = pd.concat([X_test, y_test, y_pred_proba_df, y_pred, y_cv_pred], axis=1)
result.to_csv(cnvrg_workdir+'/churn_output.csv', index=False)
############## converting arrays/lists to dataframes and printing the dataframes ################

######################## defining the composite metric and logging the metrics ###################
composite_metric_acc = acc_score-(acc_score_train-acc_score)
composite_metric_pre = pre_score-(pre_score_train-pre_score)
composite_metric_rec = rec_score-(rec_score_train-rec_score)
composite_metric_f1 = f_score-(f_score_train-f_score)

eval_metrics = [['SVC', mean(scores), acc_score, acc_score_train, pre_score,
                 pre_score_train, rec_score, rec_score_train, f_score, f_score_train]]
evaluation_df = pd.DataFrame(eval_metrics, columns=['Algo_Name', 'CV_Accuracy_Score', 'Test_Accuracy','Train_Accuracy', 'Test_Precision', 'Train_Precision', 'Test_Recall', 'Train_Recall', 'Test_F1', 'Train_F1'])

evaluation_df.to_csv(cnvrg_workdir+'/evaluation.csv', index=False)

e.log_param("accuracy_score", acc_score)
e.log_param("precision", pre_score)
e.log_param("recall_score", rec_score)
e.log_param("f1_score", f_score)
e.log_param("CV_accuracy_score", mean(scores))
e.log_param("accuracy_score_train", acc_score_train)
e.log_param("precision_train", pre_score_train)
e.log_param("recall_score_train", rec_score_train)
e.log_param("f1_score_train", f_score_train)
e.log_param("composite_metric_acc", composite_metric_acc)
e.log_param("composite_metric_pre", composite_metric_pre)
e.log_param("composite_metric_rec", composite_metric_rec)
e.log_param("composite_metric_f1", composite_metric_f1)
##################### defining the composite metric and logging the metrics ######################

################################# Confusion Matrix ###############################################
cm = pd.concat([y_test, y_pred], axis=1)
true_positive = len(cm.loc[(cm['Churn'] == 1) & (cm['Predictions'] == 1)])
true_negative = len(cm.loc[(cm['Churn'] == 0) & (cm['Predictions'] == 0)])
false_negative = len(cm.loc[(cm['Churn'] == 1) & (cm['Predictions'] == 0)])
false_positive = len(cm.loc[(cm['Churn'] == 0) & (cm['Predictions'] == 1)])

######################################## Heat Map ################################################

from cnvrgv2 import Heatmap, Cnvrg, LineChart, Experiment
e = Experiment()
cnvrg = Cnvrg()
myproj = cnvrg.projects.get("customer_churn_analysis")

heatmap_chart = Heatmap('Confusion-Matrix',
                        x_ticks=['Predicted_NotChurned', 'Predicted_Churned'],
                        y_ticks=['Actual_Churned', 'Actual_NotChurned'],
                        color_stops=[[0,'#000000'],[1, '#7EB4EB']]
                       )

heatmap_chart.add_series([(false_positive, true_positive), (true_negative, false_negative)],'s1')
e.create_chart(heatmap_chart)

########################################## ROC Curve #############################################

fpr, tpr, threshold = roc_curve(y_test['Churn'],  y_pred_proba)
auc = roc_auc_score(y_test['Churn'], y_pred_proba)
# create ROC curve
plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(cnvrg_workdir+'/ROC.png')

print('AUC is')
print(auc)
################################# Outputting Threshold ###########################################

pd.DataFrame([[threshold]], columns=['threshold']).to_csv(cnvrg_workdir+'/threshold.csv',index=False)
######################################## ROC Curve Manipulation ##################################

less_than = 0.2 # values must be less than
remove_per = 0.5 # fraction of values to be removed
def remove_values(x,y,less_than,remove_per):
    leftover_y = []
    base_rem_pool = [x for x in x if x < less_than]
    len_to_remove = round(len(base_rem_pool)*remove_per)
    to_remove = set(random.sample(range(len(base_rem_pool)),len_to_remove))    
    leftover_pool = list(set(base_rem_pool) - to_remove)
    leftover_pool.extend(list(set(x)-set(base_rem_pool)))
    for i in leftover_pool:
        leftover_y.append(y[x.index(i)])
    return leftover_pool, leftover_y
x_new, y_new = remove_values(fpr.tolist(),tpr.tolist(),less_than,remove_per)
x_new = sorted(x_new)
y_new = sorted(y_new)
loss_chart = LineChart('ROC-Curve',x_ticks=x_new)
loss_chart.add_series(y_new)
e.create_chart(loss_chart)
