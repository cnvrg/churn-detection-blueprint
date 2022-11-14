# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import os
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "-f",
    "--preprocessed_data",
    action="store",
    dest="preprocessed_data",
    default="/input/data preprocessing/data_df.csv",
    required=True,
    help="""preprocessed data""",
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

args = parser.parse_args()
preprocessed_data = args.preprocessed_data
preprocessed_data = pd.read_csv(preprocessed_data)
print(preprocessed_data.head())
# Split data into train and test sets
X = preprocessed_data.drop('Churn', axis=1)
y = preprocessed_data['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)

preprocessed_data.to_csv(cnvrg_workdir+"/preprocessed_data.csv", index=False)
X_test.to_csv(cnvrg_workdir+"/X_test.csv", index=False)
y_train.to_csv(cnvrg_workdir+"/y_train.csv", index=False)
y_test.to_csv(cnvrg_workdir+"/y_test.csv", index=False)
X_train.to_csv(cnvrg_workdir+"/X_train.csv", index=False)
