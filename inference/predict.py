import math
import random
from joblib import dump, load
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from cnvrgv2 import Cnvrg, LineChart
from numpy import mean
from cnvrg.charts import MatrixHeatmap
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict
from sklearn.feature_selection import RFECV
from cnvrg import Experiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import json
import pathlib
import sys
import requests

FILES = ['columns_list.csv','one_hot_encoder','mis_col_type.csv','my_model.sav','ordinal_enc','original_col.csv','processed_col.csv','std_scaler.bin']

BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/churn/"

def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(current_dir + f'/{f}') and not os.path.exists('/input/compare/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(BASE_FOLDER_URL + f)
            f1 = os.path.join(current_dir,f)
            with open(f1, "wb") as fb:
                fb.write(response.content)

download_model_files()

########## Path changing (whether the input comes from training or non-training ##########
if os.path.exists("/input/compare/churn_output.csv"):
    model_dir = "/input/compare/my_model.sav"
    original_col = pd.read_csv("/input/data_preprocessing/original_col.csv")
    label_encoder_path = "/input/data_preprocessing/ordinal_enc"
    oh_encoder_path = "/input/data_preprocessing/one_hot_encoder"
    processed_col = pd.read_csv("/input/data_preprocessing/processed_col.csv").columns.tolist()
    scaler = "/input/data_preprocessing/std_scaler.bin"
    columns_list_1 = pd.read_csv("/input/data_preprocessing/columns_list.csv")
    mis_col_type = pd.read_csv("/input/data_preprocessing/mis_col_type.csv")
else:
    print('Running Stand Alone Endpoint')
    script_dir = pathlib.Path(__file__).parent.resolve()
    original_col = pd.read_csv(os.path.join(script_dir,'original_col.csv'))
    columns_list_1 = pd.read_csv(os.path.join(script_dir,'columns_list.csv'))
    mis_col_type = pd.read_csv(os.path.join(script_dir,'mis_col_type.csv'))
    processed_col = pd.read_csv(os.path.join(script_dir,'processed_col.csv')).columns.tolist()
    model_dir = os.path.join(script_dir,'my_model.sav')
    scaler = os.path.join(script_dir,'std_scaler.bin')
    oh_encoder_path = os.path.join(script_dir,'one_hot_encoder')
    label_encoder_path = os.path.join(script_dir,'ordinal_enc')

###### Path changing (whether the input comes from training or non-training ######

def predict(data):
    
    threshold = 0.5
    predicted_response = {}
    cnt = 0
    predicted_response[cnt] = []
    if isinstance(data['vars'],str):
        
        data = data['vars']
        data = data.split(",")
    else:
        data = data['vars']
    
    data_new = pd.DataFrame([data])
    data_new.columns = original_col.columns[:-1]
    #################### extract id and label encoded columns ####################
    
    id_column = columns_list_1['id_columns'].dropna().tolist()
    label_encoding_columns = columns_list_1['label_encoded_columns'].dropna().tolist()
    cat_var = columns_list_1['OHE_columns'].dropna().tolist()
    ######################## extract id and label encoded columns ################
    ######################### Mising value treatment ############################
    for colname, coltype in data_new.dtypes.iteritems():
        if (pd.isna(data_new[colname][0]) == True) or (data_new[colname][0] == ''):
            test = mis_col_type.isin([colname])
            for col in test:
                if test[col].sort_values(ascending=False).unique()[0] == True:
                    if col == 'Mean':
                        data_new[colname][0] = original_col[colname][0]
                    elif col == '0-1':
                        data_new[colname][0] = random.choice([1, 0])
                    elif col == 'Median':
                        data_new[colname][0] = original_col[colname][0]
                    elif col == 'Yes-No':
                        data_new[colname][0] = random.choice(['Yes', 'No'])
                    else:
                        data_new[colname][0] = original_col[colname][0]

    data_new = data_new.replace({"'":''}, regex=True)
    for col1, col2 in original_col.dtypes.iteritems():
        if col1 != 'Churn':
            data_new[col1] = data_new[col1].astype(col2)

    ################## storing id_column values and removing id columns ##########
    id_column_list = []
    if id_column != ['None'] and id_column != []:
        for i in range(len(id_column)):
            if id_column[i] != 'None' and id_column != []:
                id_column_list.append(data_new[id_column[i]])
                data_new.drop([id_column[i]],axis=1, inplace=True)

    if label_encoding_columns != ['None'] and label_encoding_columns != []:
        garbage_index_0 = []
        for colname in label_encoding_columns:
            garbage_index = data_new.index[data_new[colname] == 'Garbage-Value-999'].tolist()
            garbage_index_0.append(garbage_index)
    ########################### Missing Value Treatment ##########################
    # Defining the map function
    def binary_map(feature):
        return feature.map({'Yes': 1, 'No': 0})
    ######################## Collating label encoded info ########################
    ################################### ONE HOT ENCODING #########################

    if cat_var != [] or os.path.exists(oh_encoder_path) == True:
        enc = load(oh_encoder_path)
        cat_var = sorted(cat_var)
        newlist = []
        for i in range(len(enc.categories_)):
            for j in range(len(enc.categories_[i])):
                newlist.append(cat_var[i]+'-'+enc.categories_[i].tolist()[j])

        temp_ohe = pd.DataFrame(enc.transform(data_new[cat_var]).toarray().tolist(),columns=newlist)
        data_new = pd.concat([data_new,temp_ohe],axis=1)
        for colname in cat_var:
            data_new = data_new.drop([colname], axis=1)
    ############################ ONE HOT ENCODING ###############################
    ########################### LABEL ENCODING ###################################
    cnt_garb = 0
    if label_encoding_columns != ['None'] or os.path.exists(label_encoder_path) == True:
        label_encoder = load(label_encoder_path)
        label_encoding_columns = sorted(label_encoding_columns)
        data_new[label_encoding_columns] = label_encoder.transform(data_new[label_encoding_columns])
        
        for colname in label_encoding_columns:
            bad_df = data_new.index.isin(garbage_index_0[cnt_garb])
            median_missing_label = data_new[~bad_df][colname].median().round()
            for j in range(len(garbage_index_0[cnt_garb])):
                data_new.at[garbage_index_0[cnt_garb][j],colname] = median_missing_label
            cnt_garb = cnt_garb+1
    ############################# LABEL ENCODING #################################
    for colname, coltype in data_new.dtypes.iteritems():
        if 'Garbage-Value-999' in colname:
            data_new = data_new.drop(colname, 1,errors='ignore')
    ##################### feature scaling #######################################
    if os.path.exists(scaler):
        sc = load(scaler)
        temp_df = pd.DataFrame()
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

    ################ feature-scaling  ######################
    one_hot_enc_cols_new = []
    num_of_cat_var = 0
    num_of_cat_var = len(cat_var)
    for colname, coltype in data_new.dtypes.iteritems():
        if coltype == 'object' and (colname not in id_column) and (colname not in cat_var) and (colname not in label_encoding_columns):
            data_new[colname] = pd.DataFrame(binary_map(data_new[colname]), columns=[colname])[colname]
    
    percentage_cat_var = num_of_cat_var/data_new.shape[1]
    if 'Churn' in processed_col:
        processed_col.remove('Churn')

    for bf in id_column:
        if bf in processed_col:
            processed_col.remove(bf)

    data_new = data_new[processed_col]
    percentage_cat_var = num_of_cat_var/data_new.shape[1]

    ######################### Loading the saved model ############################
    loaded_model = joblib.load(model_dir)
    result = np.where(loaded_model.predict_proba(data_new)[:, 1] > threshold, 1, 0)
    result_proba = loaded_model.predict_proba(data_new)[:, 1]
    y_pred = pd.DataFrame(result, columns=['Predictions'])
    ######################### Loading the saved model ###########################
    response = {}

    if id_column != ['None'] and id_column != []:
        for i in range(len(id_column)):
            response[id_column[i]] = id_column_list[i].item()
    response["Churn_Prediction"] = result.item()
    prob_value = result_proba.item()
    response["Churn_Probability"] = str(prob_value)[0:4]
    predicted_response[cnt].append(response)
    cnt=cnt+1
    return predicted_response
