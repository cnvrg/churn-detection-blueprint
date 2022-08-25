import os
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px
import pandas as pd
import numpy as np
import random
from joblib import dump, load
import joblib
from sklearn import preprocessing
import argparse
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "-f",
    "--churn_data",
    action="store",
    dest="churn_data",
    default="/data/churn_data/churn.csv",
    required=True,
    help="""churn data""",
)
parser.add_argument(
    "--label_encoding",
    action="store",
    dest="label_encoding",
    default="",
    required=True,
    help="""label encoding columns""",
)
parser.add_argument(
    "--scaler",
    action="store",
    dest="scaler",
    default=" ",
    required=True,
    help="""which scaler to use""",
)
parser.add_argument(
    "--project_dir",
    action="store",
    dest="project_dir",
    help="""--- For inner use of cnvrg.io ---""",
)
parser.add_argument(
    "--output_dir",
    action="store",
    dest="output_dir",
    help="""--- For inner use of cnvrg.io ---""",
)
parser.add_argument(
    "--id_column",
    action="store",
    dest="id_column",
    default="CustomerID",
    required=True,
    help="""id column""",
)
parser.add_argument(
    "--do_scaling",
    action="store",
    dest="do_scaling",
    default="Yes",
    required=True,
    help="""should scaling be done""",
)

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
args = parser.parse_args()
churn_data = args.churn_data
label_encoding_cols = sorted(args.label_encoding.split(','))
scaler = args.scaler
data_df = pd.read_csv(churn_data)
do_scaling = args.do_scaling
label_encoder = preprocessing.OrdinalEncoder()
id_columns = args.id_column


def dataoveriew(df, message):
    print(f'{message}:\n')
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())


no_of_rows = data_df.shape[0]
mis_val_cnt = data_df.isnull().sum().values.sum()
no_of_features = data_df.shape[1]
dataoveriew(data_df, 'Overview of the dataset')
mis_col_type = pd.DataFrame(
    columns=['Mean', '0-1', 'Median', 'Yes-No', 'String'])

#############################Mising value treatment ##############################################
original_col = data_df.head(1)
cnt_col = 0
for colname, coltype in data_df.dtypes.iteritems():

    percentage_unique = len(
        data_df[colname].dropna().unique())/len(data_df[colname].dropna())
    first_value = data_df[colname].dropna().sort_values(
        ascending=True).unique()[0]
    length_unique = len(data_df[colname].dropna().unique())
    max_unique = max(data_df[colname].dropna().unique())

    if (length_unique == 1):
        data_df.drop(colname, axis=1)
    elif ((coltype == 'int64') or (coltype == 'float64')) and (percentage_unique >= 0.1):
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = np.mean(data_df[colname])
        mis_col_type.at[cnt_col, 'Mean'] = colname
        cnt_col = cnt_col+1
        original_col[colname][0] = np.mean(data_df[colname])
    elif ((coltype == 'int64') or (coltype == 'float64')) and (max_unique < 2) and (length_unique <= 2):
        replacement = random.choices([0, 1], k=data_df[colname].isna().sum())
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = replacement
        mis_col_type.at[cnt_col, '0-1'] = colname
        cnt_col = cnt_col+1
        original_col[colname][0] = 0
    elif ((coltype == 'int64') or (coltype == 'float64')) and (percentage_unique < 0.1):
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = np.median(data_df[colname])
        mis_col_type.at[cnt_col, 'Median'] = colname
        cnt_col = cnt_col+1
        original_col[colname][0] = np.median(data_df[colname])
    elif ((first_value.lower() == 'no') or (first_value.lower() == 'yes')) and (length_unique <= 2):
        replacement = random.choices(
            ['Yes', 'No'], k=data_df[colname].isna().sum())
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = replacement
        mis_col_type.at[cnt_col, 'Yes-No'] = colname
        cnt_col = cnt_col+1
        original_col[colname][0] = 'Yes'
    else:
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = 'Garbage-Value-999'
        mis_col_type.at[cnt_col, 'String'] = colname
        cnt_col = cnt_col+1
        original_col[colname][0] = random.choice(data_df[colname].unique())

#################################### Removing label encoded missing values #######################
original_col.to_csv(cnvrg_workdir+"/original_col.csv", index=False)
garbage_index_0 = []
if label_encoding_cols != ['None']:
    for colname in label_encoding_cols:
        garbage_index = data_df.index[data_df[colname]
                                      == 'Garbage-Value-999'].tolist()
        garbage_index_0.append(garbage_index)

mis_col_type.to_csv(cnvrg_workdir+'/mis_col_type.csv', index=False)
################ Defining Binary Map function and mapping Churn column ###########################


def binary_map(feature):
    return feature.map({'Yes': 1, 'No': 0})


if data_df['Churn'][0] != 0 and data_df['Churn'][0] != 1:
    data_df['Churn'] = pd.DataFrame(binary_map(
        data_df['Churn']), columns=['Churn'])['Churn']

###################################### sparsity ##################################################
sparsity = 0
total = 0
for colname, coltype in data_df.dtypes.iteritems():
    if (coltype == 'int64') or (coltype == 'float64'):
        total = total + data_df[colname].shape[0]

sparsity = (data_df == 0).astype(int).sum(axis=1).sum()/total
##################################### charts for X-Variables #####################################
from cnvrgv2 import Cnvrg, BarChart, Experiment
cnvrg_v2 = Cnvrg()
myproj = cnvrg_v2.projects.get("customer_churn_analysis")
e_v2 = Experiment()

for colname, coltype in data_df.dtypes.iteritems():
    if ((coltype == 'int64') or (coltype == 'float64')) and (len(data_df[colname].unique()) > 0.1*data_df.shape[0]) and colname not in id_columns.split(','):
        print(colname)
        mean_1 = data_df[colname].mean()
        std_1 = data_df[colname].std()

        bin_1 = mean_1-2*std_1
        bin_2 = mean_1-1*std_1
        bin_3 = mean_1-0.5*std_1
        bin_4 = mean_1+0.5*std_1
        bin_5 = mean_1+1*std_1
        bin_6 = mean_1+2*std_1

        cnt1 = len(data_df.loc[(data_df[colname] > bin_1) & (
            data_df[colname] < bin_2) & (data_df['Churn'] == 0)])
        cnt1_a = len(data_df.loc[(data_df[colname] > bin_1) & (
            data_df[colname] < bin_2) & (data_df['Churn'] == 1)])
        cnt2 = len(data_df.loc[(data_df[colname] > bin_2) & (
            data_df[colname] < bin_3) & (data_df['Churn'] == 0)])
        cnt2_a = len(data_df.loc[(data_df[colname] > bin_2) & (
            data_df[colname] < bin_3) & (data_df['Churn'] == 1)])
        cnt3 = len(data_df.loc[(data_df[colname] > bin_3) & (
            data_df[colname] < bin_4) & (data_df['Churn'] == 0)])
        cnt3_a = len(data_df.loc[(data_df[colname] > bin_3) & (
            data_df[colname] < bin_4) & (data_df['Churn'] == 1)])
        cnt4 = len(data_df.loc[(data_df[colname] > bin_4) & (
            data_df[colname] < bin_5) & (data_df['Churn'] == 0)])
        cnt4_a = len(data_df.loc[(data_df[colname] > bin_4) & (
            data_df[colname] < bin_5) & (data_df['Churn'] == 1)])
        cnt5 = len(data_df.loc[(data_df[colname] > bin_5) & (
            data_df[colname] < bin_6) & (data_df['Churn'] == 0)])
        cnt5_a = len(data_df.loc[(data_df[colname] > bin_5) & (
            data_df[colname] < bin_6) & (data_df['Churn'] == 1)])

        x_value = ['mean-std*(2-1)', 'mean-std*(1-0.5)',
                   'mean-std*(0.5-0.5)', 'mean-std*(0.5+1)', 'mean-std*(1+2)']
        y_value1 = [cnt1, cnt2, cnt3, cnt4, cnt5]
        y_value2 = [cnt1_a, cnt2_a, cnt3_a, cnt4_a, cnt5_a]

        bar_chart = BarChart("Distribution of " + colname +
                             " by Churn Status", x_ticks=x_value)
        bar_chart.add_series(y_value1,'Not Churned')
        bar_chart.add_series(y_value2,'Churned')
        e_v2.create_chart(bar_chart)
    else:
        if colname != 'Churn' and colname not in id_columns.split(','):
            categorical_freq = data_df.groupby(
                [colname, "Churn"]).size().reset_index(name="Frequency")
            x_value = categorical_freq.loc[categorical_freq['Churn'] == 1, colname].to_list(
            )
            y_value1 = categorical_freq.loc[categorical_freq['Churn'] == 0, 'Frequency'].to_list(
            )
            y_value2 = categorical_freq.loc[categorical_freq['Churn'] == 1, 'Frequency'].to_list(
            )
            bar_chart = BarChart("Distribution of " +
                                 colname + " by Churn Status", x_ticks=x_value)
            bar_chart.add_series(y_value1, 'Not Churned')
            bar_chart.add_series(y_value2, 'Churned')
            e_v2.create_chart(bar_chart)

######################################## one hot encoding = ######################################
num_of_cat_var = 0
cat_var = []
if (scaler.lower() == 'minmax') or (scaler.lower() == 'min max') or (scaler.lower() == 'min-max') or (scaler.lower() == 'min_max'):
    sc = MinMaxScaler()
else:
    sc = StandardScaler()

for colname, coltype in data_df.dtypes.iteritems():
    if coltype == 'object' and colname not in id_columns.split(','):
        len_unique = len(data_df[colname].unique())
        first_val = data_df[colname].sort_values().unique()[0].lower()
        first_col = data_df[colname]
        if (len_unique >= 2) and (colname not in label_encoding_cols) and not (len_unique == 2 and (first_val == 'no' or first_val == 'yes')):
            num_of_cat_var = num_of_cat_var+1
            cat_var.append(colname)
        elif (len_unique == 2) and (colname not in label_encoding_cols) and (first_val == 'no' or first_val == 'yes'):
            data_df[colname] = pd.DataFrame(binary_map(
                first_col), columns=[colname])[colname]
        else:
            print(colname)
##################################### using ordinal encoder to label encode ######################
cnt_garb = 0
if label_encoding_cols != ['None']:
    label_encoder.fit(data_df[label_encoding_cols])
    ordinal_enc_path = cnvrg_workdir+'/ordinal_enc'
    dump(label_encoder, ordinal_enc_path)
    data_df[label_encoding_cols] = label_encoder.transform(
        data_df[label_encoding_cols])
    for colname in label_encoding_cols:
        bad_df = data_df.index.isin(garbage_index_0[cnt_garb])
        median_missing_label = data_df[~bad_df][colname].median().round()
        for j in range(len(garbage_index_0[cnt_garb])):
            data_df.at[garbage_index_0[cnt_garb]
                       [j], colname] = median_missing_label
        cnt_garb = cnt_garb+1

percentage_cat_var = num_of_cat_var/data_df.shape[1]

if do_scaling.lower() == 'yes':
    temp_df = pd.DataFrame()
    for colname, coltype in data_df.dtypes.iteritems():
        if (len(data_df[colname].unique())/len(data_df[colname]) > 0.1) and (coltype == 'int64' or coltype == 'float64'):
            #data_df[colname] = sc.fit_transform(data_df[[colname]])
            temp_df[colname] = data_df[colname]

    temp_df = sc.fit_transform(temp_df)
    for colname0 in temp_df:
        for colname1 in data_df:
            if(colname0 == colname1):
                data_df[colname1] = temp_df[colname0]

    dump(sc, cnvrg_workdir+'/std_scaler.bin', compress=True)

#################################### one hot encoding function ###################################


def get_encoder_inst(feature_col):
    """
    returns: an instance of sklearn OneHotEncoder fit against a (training) column feature;
    such instance is saved and can then be loaded to transform unseen data
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_df[feature_col])
    ohe_column_names = []
    for i in range(len(feature_col)):
        ohe_column_names.extend(
            (feature_col[i]+'-'+enc.categories_[i]).tolist())
    fnames = ohe_column_names
    pframe = enc.transform(data_df[feature_col]).toarray().tolist()
    OHE_df = pd.DataFrame(pframe, columns=fnames)
    file_name = cnvrg_workdir+'/one_hot_encoder'
    dump(enc, file_name)
    return OHE_df


cat_var = sorted(cat_var)
OHE_df = get_encoder_inst(cat_var)
data_df = pd.concat([data_df, OHE_df], axis=1)
for colname in cat_var:
    data_df = data_df.drop([colname], axis=1)
for colname, coltype in data_df.dtypes.iteritems():
    if 'Garbage-Value-999' in colname:
        data_df = data_df.drop(colname, 1, errors='ignore')
data_df.to_csv(cnvrg_workdir+"/data_df.csv", index=False)
data_df.head(1).to_csv(cnvrg_workdir+"/processed_col.csv", index=False)

#######################################no_of_features ############################################
dimensionality_ratio = data_df.shape[0]/data_df.shape[1]
######################################### no_of_features #########################################

##################################### compiling information ######################################
recommendation = pd.DataFrame(columns=[
                              'Sparsity', 'Categorical_Percentage', 'Dimensionality', 'Recommendation', 'General Comment'])

for ix in range(1):
    recommendation.at[ix, 'Sparsity'] = sparsity
    recommendation.at[ix, 'Dimensionality'] = dimensionality_ratio
    recommendation.at[ix, 'Categorical_Percentage'] = percentage_cat_var
    if sparsity > 0.5 and percentage_cat_var < 0.25:
        recommendation.at[ix, 'Recommendation'] = 'Logistic Regression'
        recommendation.at[ix,
                          'General Comment'] = 'Sparsity is high and categorical features are low'
    elif percentage_cat_var < 0.15:
        recommendation.at[ix, 'Recommendation'] = 'Random Forest'
    elif dimensionality_ratio < 1:
        recommendation.at[ix, 'Recommendation'] = 'Support Vector Machine'
        recommendation.at[ix,
                          'General Comment'] = 'Either decrease features or increase observations'
    elif percentage_cat_var > 0.5:
        recommendation.at[ix, 'Recommendation'] = 'Naive Bayes'
        recommendation.at[ix,
                          'General Comment'] = 'High number of categorical variables'
    else:
        recommendation.at[ix, 'Recommendation'] = 'Any'
        recommendation.at[ix, 'General Comment'] = 'No comment'

recommendation.to_csv(cnvrg_workdir+'/recommendation.csv', index=False)

################################### compiling list of col types ##################################

if (len(id_columns.split(',')) > len(label_encoding_cols) and len(id_columns.split(',')) > len(cat_var)):
    columns_list = pd.DataFrame(columns=['id_columns'])
    columns_list['id_columns'] = id_columns.split(',')
    columns_list = pd.concat([columns_list, pd.DataFrame(
        label_encoding_cols, columns=['label_encoded_columns'])], axis=1)
    columns_list = pd.concat(
        [columns_list, pd.DataFrame(cat_var, columns=['OHE_columns'])])
elif len(label_encoding_cols) > len(cat_var):
    columns_list = pd.DataFrame(columns=['label_encoded_columns'])
    columns_list['label_encoded_columns'] = label_encoding_cols
    columns_list = pd.concat([columns_list, pd.DataFrame(
        id_columns.split(','), columns=['id_columns'])], axis=1)
    columns_list = pd.concat([columns_list, pd.DataFrame(
        cat_var, columns=['OHE_columns'])], axis=1)
else:
    columns_list = pd.DataFrame(columns=['OHE_columns'])
    columns_list['OHE_columns'] = cat_var
    columns_list = pd.concat([columns_list, pd.DataFrame(
        id_columns.split(','), columns=['id_columns'])], axis=1)
    columns_list = pd.concat([columns_list, pd.DataFrame(
        label_encoding_cols, columns=['label_encoded_columns'])], axis=1)

columns_list.to_csv(cnvrg_workdir+'/columns_list.csv', index=False)
################################ Logging metrics #################################################
from cnvrg import Experiment
cnvrg = Cnvrg()
e = Experiment()
e.log_param("sparsity", sparsity)
e.log_param("dimensionality_ratio", dimensionality_ratio)
e.log_param("Categorical_Percentage", percentage_cat_var)
e.log_param("Number-of-rows", no_of_rows)
e.log_param("Missing value count", mis_val_cnt)
e.log_param("Number of features", no_of_features)
