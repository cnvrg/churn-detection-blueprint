# Train Test Split for Chrun Analysis
- Splitting the data into train and test samples.

## Features

- read the preprocessed data csv from the code atrifacts
- split the data into independent and dependant variables
- perform a train test split of the data
- store the resultants in the data artifacts to be used in the algorithms

## Input Arguments
- `--preprocessed_data` preprocessed and clean data from the data preprocessing block

## Model Artifacts
- `--X_train.csv` train dataset of independent variables
- `--X_test.csv` test dataset of independent variables
- `--y_train.csv` train dataset of dependent variables
- `--y_test.csv` test dataset of dependent variables

How to run
```
python3 train_test_split/tts.py --preprocessed_data /input/data_preprocessing/data_df.csv 
```

## About Train Test Split
Train Test Split library serves as a module for a stratified splitting of the data into test and train samples to make sure that each user’s information is split proportionally into both test and train datasets