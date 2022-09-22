from joblib import load
from sklearn import preprocessing
import os
import random
import joblib
import matplotlib.pyplot as plt
from numpy import mean
import argparse
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "-f",
    "--datafile",
    action="store",
    dest="datafile",
    default="/data/training_data.csv",
    required=True,
    help="""independent variables data, to use the model on""",
)
parser.add_argument(
    "--model_dir",
    action="store",
    dest="model_dir",
    default="/data/saved_model/",
    required=True,
    help="""model pickle file that has been trained """,
)
parser.add_argument(
    "--oh_encoder",
    action="store",
    dest="oh_encoder",
    default="/data/churn_data/one_hot_encoder",
    required=True,
    help="""ohe saved from training module""",
)
parser.add_argument(
    "--label_encoder_file",
     action="store",
    dest="label_encoder_file",
    default="/data/churn_data/ordinal_enc.joblib",
    required=True,
    help="""ordinal encoder file """,
)
parser.add_argument(
    "--columns_list",
    action="store",
    dest="columns_list",
    default="/data/churn_data/columns_list.csv",
    required=True,
    help="""columns that have been label encoded and those columns which are ID """,
)
parser.add_argument(
    "--scaler",
    action="store",
    dest="scaler",
    default="/data/churn_data/std_scaler.bin",
    required=True,
    help="""scaler loaded after being saved from training """,
)
parser.add_argument(
    "--threshold",
    action="store",
    dest="threshold",
    default="0.5",
    required=True,
    help="""threshold above which the churn is taken to be true""",
)
parser.add_argument(
    "--processed_file_col",
    action="store",
    dest="processed_file_col",
    default="/data/churn_data/processed_col",
    required=True,
    help="""column order of the data preprocessing blueprint output""",
)
parser.add_argument(
    "--do_scaling",
    action="store",
    dest="do_scaling",
    default="Yes",
    required=True,
    help="""flag whether to perform scaling operation on the dataset or not""",
)
parser.add_argument(
    "--mis_col_type",
    action="store",
    dest="mis_col_type",
    default="Yes",
    required=True,
    help="""colum which shows what kind of missing value treatment was done to specific columns""",
)


cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
args = parser.parse_args()
data_new = pd.read_csv(args.datafile)
model_dir = args.model_dir
columns_list_1 = pd.read_csv(args.columns_list)
scaler = args.scaler
threshold = float(args.threshold)
processed_file_col = pd.read_csv(args.processed_file_col).columns.tolist()
id_column = columns_list_1['id_columns'].dropna().tolist()
label_encoding_columns = columns_list_1['label_encoded_columns'].dropna().tolist()
do_scaling = args.do_scaling
mis_col_type = pd.read_csv(args.mis_col_type)
###### Data Summarization - Removal of ID Column - Mapping of Dependent Variable ######

def dataoveriew(df, message):
    print(f'{message}:\n')
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())

no_of_rows = data_new.shape[0]
mis_val_cnt = data_new.isnull().sum().values.sum()
no_of_features = data_new.shape[1]
dimensionality_ratio = data_new.shape[0]/data_new.shape[1]

dataoveriew(data_new, 'Overview of the dataset')

######################### Dropping customer ID column and storing it ##################
id_column_list = []
if id_column != ['None'] and id_column != []:
    for i in range(len(id_column)):
        id_column_list.append(data_new[id_column[i]])
        data_new.drop([id_column[i]], axis=1, inplace=True)

############################# Mising value treatment ##################################
for colname, coltype in data_new.dtypes.iteritems():
    percentage_unique = len(data_new[colname].dropna().unique())/len(data_new[colname])
    first_value = data_new[colname].dropna().sort_values(ascending=True).unique()[0]
    length_unique = len(data_new[colname].dropna().unique())
    max_unique = max(data_new[colname].dropna().unique())
    
    if (length_unique == 1):
        data_new.drop(colname, axis=1)
    elif ((coltype == 'int64') or (coltype == 'float64')) and (percentage_unique > 0.1):
        nans = data_new[colname].isna()
        data_new.loc[nans, colname] = np.mean(data_new[colname])
    elif ((coltype == 'int64') or (coltype == 'float64')) and (max_unique < 2) and (length_unique <= 2):
        replacement = random.choices([0, 1], k=data_new[colname].isna().sum())
        nans = data_new[colname].isna()
        data_new.loc[nans, colname] = replacement
    elif ((coltype == 'int64') or (coltype == 'float64')) and (percentage_unique < 0.1):
        nans = data_new[colname].isna()
        data_new.loc[nans, colname] = np.median(data_new[colname])
    elif ((first_value.lower() == 'no') or (first_value.lower() == 'yes')) and (length_unique <= 2):
        replacement = random.choices(['Yes', 'No'], k=data_new[colname].isna().sum())
        nans = data_new[colname].isna()
        data_new.loc[nans, colname] = replacement
    else:
        nans = data_new[colname].isna()
        data_new.loc[nans, colname] = 'Garbage-Value-999'

if label_encoding_columns != ['None']:
    garbage_index_0 = []
    for colname in label_encoding_columns:
        garbage_index = data_new.index[data_new[colname] == 'Garbage-Value-999'].tolist()
        garbage_index_0.append(garbage_index)

############################# defining binary map function ############################
def binary_map(feature):
    return feature.map({'Yes': 1, 'No': 0})


###################################### sparsity #######################################
sparsity = 0
total = 0
for colname, coltype in data_new.dtypes.iteritems():
    if (coltype == 'int64') or (coltype == 'float64'):
        total = total + data_new[colname].shape[0]

sparsity = (data_new == 0).astype(int).sum(axis=1).sum()/total
######################################### sparsity ####################################

cat_var = []
cat_var = columns_list_1['OHE_columns'].dropna().tolist()
num_of_cat_var = len(cat_var)
for colname, coltype in data_new.dtypes.iteritems():
    if coltype == 'object' and (colname not in id_column) and (colname not in cat_var) and (colname not in label_encoding_columns):
        data_new[colname] = pd.DataFrame(binary_map(data_new[colname]), columns=[colname])[colname]

################################### ONE HOT ENCODING ##################################
if os.path.exists(args.oh_encoder):
    
    if cat_var != [] or os.path.exists(args.oh_encoder) == True:
        enc = load(args.oh_encoder)
        cat_var = sorted(cat_var)
        newlist = []
        for i in range(len(enc.categories_)):
            for j in range(len(enc.categories_[i])):
                newlist.append(cat_var[i]+'-'+enc.categories_[i].tolist()[j])

        temp_ohe = pd.DataFrame(enc.transform(data_new[cat_var]).toarray().tolist(),columns=newlist)
        data_new = pd.concat([data_new,temp_ohe],axis=1)
        for colname in cat_var:
            data_new = data_new.drop([colname], axis=1)

################################### LABEL ENCODING ####################################
cnt_garb = 0
if os.path.exists(args.label_encoder_file):
    
    if label_encoding_columns != ['None']:
        label_encoder = load(args.label_encoder_file)
        label_encoding_columns = sorted(label_encoding_columns)
        data_new[label_encoding_columns] = label_encoder.transform(data_new[label_encoding_columns])
        for colname in label_encoding_columns:
            bad_df = data_new.index.isin(garbage_index_0[cnt_garb])
            median_missing_label = data_new[~bad_df][colname].median().round()
            for j in range(len(garbage_index_0[cnt_garb])):
                data_new.at[garbage_index_0[cnt_garb][j],colname] = median_missing_label
            cnt_garb = cnt_garb+1

for colname, coltype in data_new.dtypes.iteritems():
    if 'Garbage-Value-999' in colname:
        data_new = data_new.drop(colname, 1,errors='ignore')
    
################################### reordering data ###################################

percentage_cat_var = num_of_cat_var/data_new.shape[1]
processed_file_col.remove('Churn')

for bf in id_column:
    if bf in processed_file_col:
        processed_file_col.remove(bf)

data_new = data_new[processed_file_col]

percentage_cat_var = num_of_cat_var/data_new.shape[1]
################################### feature scaling ###################################
if os.path.exists(scaler):
    
    sc = load(scaler)
    temp_df = pd.DataFrame()

    if do_scaling.lower() == 'yes':
        for colname, coltype in data_new.dtypes.iteritems():
            test1 = mis_col_type.isin([colname])
            for col in test1:
                if test1[col].sort_values(ascending=False).unique()[0] == True:
                    if col == 'Mean':
                        temp_df[colname] = data_new[colname]
        temp_df = sc.transform(temp_df)
        for colname0 in temp_df:
            for colname1 in data_new:
                if(colname0 == colname1):
                    data_new[colname1] = temp_df[colname0]
########################### one hot encoding of new dataset ###########################

############################## Loading the saved model ################################
loaded_model = joblib.load(model_dir)
result = np.where(loaded_model.predict_proba(data_new)[:, 1] > threshold, 1, 0)
result_proba = loaded_model.predict_proba(data_new)[:, 1]

if id_column != ['None'] and id_column != []:
    for i in range(len(id_column)):
        data_new[id_column[i]] = id_column_list[i]
result_proba = [str(x)[0:4] for x in result_proba]
y_pred = pd.DataFrame(result, columns=['Predictions'])
y_proba = pd.DataFrame(result_proba, columns=['Prediction Percentage'])
final_result = pd.concat([data_new, y_pred], axis=1)
final_result = pd.concat([final_result, y_proba], axis=1)
final_result.to_csv(cnvrg_workdir+'/churn.csv', index=False)
################################ Loading the saved model ##############################
