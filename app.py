import os
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse

import mlflow

import mlflow.sklearn
import logging
import os

os.environ["MLFLOW_TRACKING_URI"]="http://ec2-[your_ec2_public_dns].ap-southeast-2[your_region].compute.amazonaws.com:5000/"

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, preds):
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)

    return rmse, mae, r2


if __name__ == "__main__":
        ## For the remote server AWS
    remote_server_uri = "http://ec2-[your_ec2_public_dns].ap-southeast-2[your_region].compute.amazonaws.com:5000/"

    mlflow.set_tracking_uri(remote_server_uri)


    ## Data Ingestion: reading the wine quality dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download the data")


    ## Split the dataset:
    train, test= train_test_split(data, random_state=1, test_size=0.2)
    
    X_train = train.drop(['quality'], axis=1)
    y_train = train[['quality']]
    X_test = test.drop(['quality'], axis=1)
    y_test = test[['quality']]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    mlflow.set_experiment("ElasticNet-Wine-S3")
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ration={:f})".format(alpha, l1_ratio))
        print(" RMSE: %s" % rmse)
        print(" MAE: %s" % mae)
        print(" R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ration", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticNetWineModel"
            )
        else:
            mlflow.sklearn.log_model(
                lr, "model"
            )





