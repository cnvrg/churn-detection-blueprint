import pandas as pd
import os
import psutil
import time
from cnvrg import Experiment
import shutil

tic = time.time()

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

for k in os.environ.keys():
    if 'PASSED_CONDITION' in k and os.environ[k] == 'true':
        print("Yes123")
        task_name = k.replace('CNVRG_', '').replace(
            '_PASSED_CONDITION', '').lower()
train_file = '/input/'+task_name+'/'+'churn_output.csv'
train = pd.read_csv(train_file)
file_name_variable = 'churn_output.csv'
train.to_csv(cnvrg_workdir+"/{}".format(file_name_variable), index=False)

model = '/input/'+task_name+'/'+'my_model.sav'
destination = cnvrg_workdir+"/my_model.sav"
shutil.copy(model, destination)

eval_file = '/input/'+task_name+'/'+'evaluation.csv'
eval_1 = pd.read_csv(eval_file)
file_name_variable_0 = 'evaluation.csv'

eval_1['overfit_magnitude'] = eval_1['Test_Accuracy']-eval_1['Train_Accuracy']
overfit_magnitude = eval_1['Test_Accuracy']-eval_1['Train_Accuracy']

eval_1['avg_train_metric'] = eval_1[['Train_Accuracy','Train_Precision','Train_Recall','Train_F1']].mean(axis=1)

avg_train_metric = eval_1[['Train_Accuracy','Train_Precision','Train_Recall','Train_F1']].mean(axis=1)

avg_test_metric = eval_1[['Test_Accuracy','Test_Precision','Test_Recall','Test_F1']].mean(axis=1)

eval_1['avg_test_metric'] = eval_1[['Test_Accuracy','Test_Precision','Test_Recall','Test_F1']].mean(axis=1)

eval_1['pre_rec_metric'] = (eval_1['Test_Precision']-eval_1['Test_Recall'])/eval_1['Test_Recall']
prec_rec_metric = (eval_1['Test_Precision']-eval_1['Test_Recall'])/eval_1['Test_Recall']

eval_1['Comment'] = 'xyz'
for i in range(eval_1.shape[0]):
    if eval_1.at[i, 'avg_test_metric'] > eval_1.at[i, 'avg_train_metric']:
        eval_1.at[i, 'Comment'] = 'Low Variance in Data or Too few samples'
    else:
        eval_1.at[i, 'Comment'] = 'None'
    if eval_1.at[i, 'pre_rec_metric'] > 0.5:
        eval_1.at[i, 'Comment_2'] = 'too few churn cases and/or high accuracy when predicting churn'
    if eval_1.at[i, 'pre_rec_metric'] < -0.5:
        eval_1.at[i, 'Comment_2'] = 'too many churn cases and/or high accuracy when predicting non-churn'
    else:
        eval_1.at[i, 'Comment_2'] = 'reasonably consistent predictions and data'
eval_1.to_csv(cnvrg_workdir+"/{}".format(file_name_variable_0), index=False)

print('RAM GB used:', psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
toc = time.time()
print("time taken:", toc-tic)
e = Experiment()
e.log_param("compare_ram", psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
e.log_param("compare_time", toc-tic)

e.log_param("overfit magnitude",overfit_magnitude[0])
e.log_param("prec-rec-metric",prec_rec_metric[0])
e.log_param("avg_train_metric",avg_train_metric[0])
e.log_param("avg_test_metric",avg_test_metric[0])


#from cnvrgv2 import Experiment
#e = Experiment()
#cnvrg = Cnvrg()
#myproj = cnvrg.projects.get("customer_churn_analysis")
#e.log("overfit magnitude",overfit_magnitude)
#e.log("prec-rec-metric",prec_rec_metric)
#e.log("avg_train_metric",avg_train_metric)
#e.log("avg_test_metric",avg_test_metric)