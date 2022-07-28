You can use this blueprint to clean and validate data in order to further train multiple models to predict results for if a customer is likely to churn or not using your own customized dataset. In order to clean the data you will be needed to provide:
`--churn_data` raw data uploaded by the user on the platform
`--id_column` get the name of the id column by the user
`--label_encoding_cols` list of columns to be label encoded by the user
`--scaler` the type of scaler to be used
`--do_scaling` whether scaling will be done or not 

Directions for use:
- Click on Use Blueprint button
- In the pop up, choose the relevant compute you want to use to deploy your API endpoint
- You will be redirected to your endpoint
- You can now use the Try it Live section with any image (for example: LINK)
- You can now integrate your API with your code using the integration panel at the bottom of the page
- You will now have a functioning API endpoint that predicts customer churn