##############################################
##############################################
################   IMPORTS   #################
#########       ONLY THE BEST       ##########
##############################################
##############################################

import json
import smtplib
import os
import secrets
import logging

import pandas as pd
import numpy as np
import pyspark
from spark_sklearn import GridSearchCV as SparkGridSearchCV

# import tf_tools # ( •_•)>⌐■-■ (⌐■_■)

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, when
from pyspark.sql.types import StructType

from sklearn.feature_selection import RFECV
# from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,auc,precision_recall_curve,recall_score,roc_curve,roc_auc_score,precision_score,confusion_matrix,classification_report

import lightgbm.sklearn as lgb
from sklearn.externals import joblib

logging.basicConfig(filename="generate_model.log", level=logging.INFO)

##############################################
##############################################
#########        CONFIG          #############
##############################################
##############################################


logging.info("Setting config values.")

# spark stuff
spark_executor_memory = '64g' #string
spark_executor_cores = '8' #string

training_data_path = ""
model_ser_path = '.pkl'

id_cols = []

target_cols = []

fields_config_file= ""


param_grid = {'boosting_type': ['dart']
#              ,'n_estimators': [10,100,200]
             ,'learning_rate':[0.01,0.1,0.3]
             ,'num_leaves': [10,100,200]
#              ,'max_depth': [10,50]
#              ,'min_child_samples':[20,100,200]
             }

model_name = ""
target_var = ''

ear_stop_eval_mtr = 'auc'
ear_stop_rnds = 25


##############################################
##############################################
##########  MODEL GENERATION FUNCTION   ######
##############################################
##############################################

def generate_model_package(
            training_data_path
            ,id_cols
            ,target_cols
            ,fields_config_file
            ,param_grid
            ,model_name
            ,target_var
        ):

    """
            training_data_path
            ,id_cols
            ,target_cols
            ,fields_config_file
            ,param_grid
            ,model_name
            ,target_var
    """

    pyspark_app_nm = "train_" + model_name + "_" + secrets.token_hex(nbytes=4)

    logging.info("Starting process: " + pyspark_app_nm)

    #create spark object and spark context for parallel learning
    logging.info("Instantiating pyspark.")
    app_pyspark_conf = SparkConf()
    app_pyspark_conf.setAppName(pyspark_app_nm)
#     app_pyspark_conf.set('spark.executor.memory',spark_executor_memory)
#     app_pyspark_conf.set('spark.executor.cores', spark_executor_cores)

    spark = SparkSession.builder.config(conf=app_pyspark_conf).getOrCreate()
    sc = spark.sparkContext


    #load data
    logging.info("Beginning data load.")
    training_df = pd.read_parquet(training_data_path,engine='pyarrow')
    # sampling down
#     training_df_1 = training_df[training_df[target_var]==1].sample(20)
#     training_df_0 = training_df[training_df[target_var]==0].sample(40)
#     training_df = pd.concat([training_df_0,training_df_1])

    # column handling
    logging.info("Creating column lists")
    all_cols= training_df.columns.tolist()
    x_cols = list(set(all_cols) - (set(target_cols + id_cols)))

    # dataframe setup
    X = training_df[x_cols]
    y = training_df[target_cols]

    # create holdout data
    logging.info("Creating holdout data")
    x_train, x_test, y_train, y_test = train_test_split(X, y[target_var], test_size=0.1, stratify=y[target_var])

    wts= y_test.value_counts()
    wtrat=(wts[0]/wts[1])

    # instantiate model
    gbm = lgb.LGBMClassifier()

    fit_params = {
        "eval_set":[(x_test, y_test)]
        ,"eval_metric": ear_stop_eval_mtr
        ,"early_stopping_rounds": ear_stop_rnds
#         ,"scale_pos_weight": wtrat
    }

    grid_search = SparkGridSearchCV(sc,estimator=gbm,param_grid=param_grid, fit_params=fit_params)
#     grid_search.fit(x_train,y_train)


    grid_search.fit(x_train,y_train)


    best_model = grid_search.best_estimator_
    optimized_parameters = best_model.get_params()

    # create confusion dataframe
    y_true = pd.DataFrame(y_test)
    y_true = y_true.reset_index()
    y_true.columns.values[0] = "CUSTOMER_KEY"
    y_true.columns.values[1] = "Y_TRUE"

    y_pred = pd.DataFrame(best_model.predict(x_test,y_test.tolist())
                          ,columns=["Y_PRED"])

    confusion_data = pd.merge(
        left=y_true
        ,right=y_pred
        ,left_index=True
        ,right_index=True)

    # summary statistics and metrics

    fr_col_nam_map = {0:"feature_nm",1:"feature_importance"}
    feature_ranking = pd.DataFrame([X.columns,best_model.feature_importances_]).T
    feature_ranking = feature_ranking.rename(columns=fr_col_nam_map)
    feature_ranking = feature_ranking.sort_values("feature_nm",ascending=False)




    metrics = {
        "precision_score": precision_score(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"roc_auc_score": roc_auc_score(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"classification_report": classification_report(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"confusion_matrix": confusion_matrix(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"accuracy_score": accuracy_score(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"precision_recall_curve": precision_recall_curve(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"recall_score": recall_score(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
        ,"roc_curve": roc_curve(confusion_data['Y_TRUE'],confusion_data['Y_PRED'])
            }

    output = {
                "model_name": model_name # string with model name
                ,"model_class": best_model # grid_search.best_estimator_
                ,"optimized_parameters": optimized_parameters # best_model.get_params()
                ,"feature_ranking": feature_ranking # best_model.feature_importances_
                ,"metrics": metrics
                ,"confusion_data": confusion_data
             }

    return output



##############################################
############    INVOCATION    ################
##############################################

test_model_pak = generate_model_package(
    training_data_path = training_data_path
    ,id_cols = id_cols
    ,target_cols = target_cols
    ,fields_config_file = fields_config_file
    ,param_grid = param_grid
    ,model_name = model_name
    ,target_var=target_var)


##############################################
###########    SERIALIZATION    ##############
##############################################

joblib.dump(test_model_pak, model_ser_path)
