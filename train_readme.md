You can use this blueprint to clean and validate data in order to further train multiple models to predict results for if a customer is likely to churn or not using your own customized dataset. In order to clean the data you will be needed to provide:
`--churn_data` raw data uploaded by the user on the platform
`--id_column` get the name of the id column by the user
`--label_encoding_cols` list of columns to be label encoded by the user
`--scaler` the type of scaler to be used
`--do_scaling` whether scaling will be done or not 

You would need to provide 1 folder in s3 where you can keep your training data
- churn_data: Folder containing the training data "churn.csv" and the test data file on which to predict, "data_new.csv"

Directions for use:
1. Click on Use Blueprint button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the input file is located

   In the `Data-Preprocessing` task:
    *  Under the `churn_data` parameter provide the path to the input folder including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/churn_data`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will train a new churn detection model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the `Try it Live` section with a data point similar to your input data (in terms of variables and data types) to check your model
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that detects about-to-churn customers amongst the total list of customers!

[See here how we created this blueprint](https://github.com/cnvrg/churn-detection-blueprint)
