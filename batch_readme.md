You can use this blueprint to predict the probability of a customer churning. You can replace the model and data preprocessing artifacts (scalers, encoders etc) in the s3 bucket "libhub-readme" under the directory "churn_data".

In order to run this blueprint, you need to provide a list files (artifacts and test file) and upload them in s3 and accordingly modify the parameters in yaml file:
- churn_data
    -churn.csv
    -columns_list.csv
    -mis_col_type.csv
    -my_model.sav
    -one_hot_encoder
    -ordinal_enc
    -processed_col.csv
    -std_scaler.bin

1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data (libhub readme)
    * Under the `prefix` parameter provide the main path to where the images are located
   In the `Batch-Predict` task:
    *  Under the `datafile` parameter provide the path to the directory including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/churn_data/`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will deploy a churn detection model and predict the probability of a customer being churned and download the CSV file with the information about the scenery. Go to output artifacts and check for the output csv file.

Congrats! You have deployed a custom model that predicts the probability of a customer churning!

[See here how we created this blueprint](https://github.com/cnvrg/churn-detection-blueprint)