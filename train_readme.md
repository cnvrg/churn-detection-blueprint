Use this blueprint to clean and validate customized data, train multiple models using a preprocessed dataset, and deploy a new API endpoint that predicts whether a customer or client is likely to churn. To train this model with your data, provide one `churn_data` folder in the S3 Connector to store the data, which contains the `churn.csv` training data file and the `data_new.csv` test data file on which to make predictions.

Complete the following steps to train the churn-detector model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `bucketname` - Value: enter the data bucket name
     * Key: `prefix` - Value: provide the main path to the CSV file folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Data Preprocessing** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `churn_data` – Value: provide the path to the CSV file including the S3 prefix
     * `/input/s3_connector/<prefix>/churn_data` − ensure the CSV file path adheres this format
     NOTE: You can use prebuilt data example paths already provided.
   * Click the **Advanced** tab to change the resources to run the blueprint, as required.
4. Click the **Train Test Split** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
      * Key: `--preprocessed_data` – Value: the path to the CSV file from the Data Preprocessing task
	   * `/input/data_preprocessing/data_df.csv` − ensure the data path adheres this format
      NOTE: You can use prebuilt data example paths already provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
5. Click the **Compare** task to display its dialog.
   * Click the **Conditions** tab to set the following metric conditions information, as required:
     * `--accuracy_score` − set the accuracy, either the fraction (default) or the count (normalize=false) of correct predictions
     * `--recall_score` − set the model’s performance (sensitivity) based on the correct classification percentages of relevant results; a value of 1 is considered a high, ideal classifier
     * `--precision` − set the model’s performance based on the percentage of pertinent results; a value of 1 is considered a high, good classifier
     * `--f1_score` − set this statistical measure to rate the model's performance. Defined as the harmonic mean of precision and recall, the F1 score is considered a better measure than accuracy. The F1 score becomes 1 only when precision and recall are both 1.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
6. Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained churn-detector model and deploying it as a new API endpoint.
7. Track the blueprint's real-time progress in its experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
8. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   * Use the Try it Live section with a relevant user ID to check the model’s prediction accuracy.
   * Use the bottom integration panel to integrate your API with your code by copying in your code snippet.
   
A custom model and an API endpoint which detect about-to-churn customers among a larger customer pool have now been trained and deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/churn-detection-blueprint).
