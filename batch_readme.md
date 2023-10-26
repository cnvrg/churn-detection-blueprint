Use this blueprint to run in batch mode a pretrained tailored model that predicts whether a customer or client is likely to churn. The model can be trained using this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/churn-detection-train), after which the trained model can be uploaded to the S3 Connector.

To run this blueprint, replace the model (optional) and data preprocessing artifacts (such as scalers and encoders) in the S3 Connector’s `churn_data` directory by providing a list of files (artifacts and test file) and modifying the parameters in YAML file accordingly.

Complete the following steps to run this churn-detector blueprint in batch mode:
1. Click **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` − Value: provide the data bucket name
     - Key: `prefix` – Value: provide the main path to the churn data folder
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Click the **Batch Predict** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `datafile` − Value: provide the S3 path to the churn data file in the following format: ` /input/s3_connector/churn_data/churn.csv`
     - Key: `model_dir` − Value: provide the S3 pagh to the saved model in the following format: `/input/s3_connector/churn_data/my_model.sav`
     - Key: `scaler` − Value: provide the S3 path to the scaler in the following format: `/input/s3_connector/churn_data/std_scaler.bin`
     - Key: `columns_list` − Value: provide the S3 path to the columns list in the following format: `/input/s3_connector/churn_data/columns_list.csv`
     - Key: `oh_encoder` − Value: provide the S3 path to the one hot encoder in the following format: `/input/s3_connector/churn_data/one_hot_encoder`
     - Key: `label_encoder_file` − Value: provide the S3 path to the label encoder file in the following format: `/input/s3_connector/churn_data/ordinal_enc`
     - Key: `processed_file_col` − Value: provide the S3 path to the processed file column in the following format: ` /input/s3_connector/churn_data/processed_col.csv`
     - Key: ` mis_col_type` − Value: provide the S3 path to the missed column type in the following format: `/input/s3_connector/churn_data/mis_col_type.csv`
     NOTE: You can use prebuilt data example paths provided.
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software deploys a churn-detector model that predicts churn likelihood probabilities for customers and clients.
5. Track the blueprint’s real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Select **Batch Inference > Experiments > Artifacts** and locate the batch output CSV file.
7. Select the **churn.csv** File Name, click the Menu icon, and select **Open File** to view the output CSV file.

A custom churn-detector model that can predict churn likelihood probabilities for customers and clients has now been deployed in batch mode. To learn how this blueprint was created, click [here](https://github.com/cnvrg/churn-detection-blueprint).
