import json
import smtplib
import os
import pandas as pd
import numpy as np

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, when

import tf_tools

pyspark_app_nm = "pyspark_testing"
trainer_df_prqt_path = ""
null_fields_list = []


spark = SparkSession.builder.appName(pyspark_app_nm).getOrCreate()

trainer_df = spark.read.parquet(trainer_df_prqt_path)

cols_num, cols_str,col_types = tf_tools.sort_cols(trainer_df)


modes_master = tf_tools.train_categorical_imputer(trainer_df,cols_str)

trainer_df = tf_tools.apply_categorical_imputer(trainer_df,cols_str,modes_master,True)

cols_num = list(set(cols_num) - set(null_fields_list))

means_master = tf_tools.train_numeric_imputer(trainer_df,cols_num)

numputed_df = tf_tools.apply_numeric_imputer(trainer_df,cols_num,means_master)

encoder_vals = tf_tools.train_encoder(numputed_df,cols_str)

trainer_df = tf_tools.apply_encoder(numputed_df,encoder_vals)

scaler_values = tf_tools.train_scaler(trainer_df,cols_num)

scaled_values = tf_tools.apply_scaler(trainer_df,cols_num,scaler_values)
