# imports - only the best imports
import json
import smtplib
import os
import pandas as pd
import numpy as np
import pyspark
import datetime

import tf_tools # ( •_•)>⌐■-■ (⌐■_■)

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, when
from pyspark.sql.types import StructType


# Prep log file
file_ts = datetime.datetime.now().strftime("%Y%m")
logf = open("generate_training_data_{dt}.log".format(dt=file_ts),"w")


# Config

pyspark_app_nm = ""

input_data_path = "/opt/app/modeler_data/skinny_car_201801.csv"
input_targets_path = "/opt/app/modeler_data/target_file.parquet"
validation_data_path = "/opt/app/modeler_data/skinny_car_201802.csv"
training_data_write_path = "/opt/app/modeler_data/training_file2.parquet"

fields_config_file="sk_col_handler.csv"

master_id = ["CUSTOMER_KEY"]
join_id = "CUSTOMER_KEY"

drop_list = []

problem_fields = []

# spark config/instantiate spark

app_pyspark_conf = SparkConf()
app_pyspark_conf.setAppName(pyspark_app_nm)
# app_pyspark_conf.set('spark.executor.memory','100g')
# app_pyspark_conf.set('spark.executor.cores', '12')
# app_pyspark_conf.set("spark.network.timeout","3600000")
# app_pyspark_conf.set('spark.driver.memory', '45G')

spark = SparkSession.builder.config(conf=app_pyspark_conf).getOrCreate()



# load data from csv path
input_df = spark.read.csv(
    input_data_path
    ,inferSchema = True
    ,header=True)


# column handling
col_handler = pd.read_csv(fields_config_file)

id_cols = col_handler['field_nm'][col_handler['type']=='id'].tolist()
cat_cols = col_handler['field_nm'][col_handler['type']=='cat'].tolist()
flag_default_false_cols = col_handler['field_nm'][col_handler['type']=='flag_default_false'].tolist()
flag_default_null_cols = col_handler['field_nm'][col_handler['type']=='flag_default_null'].tolist()
flag_yn_cols = col_handler['field_nm'][col_handler['type']=='flag_yn'].tolist()

num_cols = col_handler['field_nm'][col_handler['type']=='num'].tolist()
num_model_cols = col_handler['field_nm'][col_handler['type']=='num_model'].tolist()
num_model_cols = list(set(num_model_cols)- set(problem_fields))


# load target data
targets_df = spark.read.parquet(
    input_targets_path)
target_cols = targets_df.columns
target_cols.remove("__index_level_0__")


# join X to Y
training_df = input_df.join(
    other = targets_df
    ,on = join_id
    ,how = "left")

# mark null flags as false
training_df = training_df.fillna(0,subset=flag_default_false_cols+flag_default_null_cols)

# fix fields with y/n instead of 1/0
for i in flag_yn_cols:
    training_df = training_df.withColumn("flg_"+i,when(training_df[i]=="Y",1).otherwise(0))

# impute categorical values
cat_imp_rules = tf_tools.train_categorical_imputer(training_df,cat_cols)
training_df = tf_tools.apply_categorical_imputer(training_df,cat_cols,cat_imp_rules,True)

# impute numeric values
num_imp_rules = tf_tools.train_numeric_imputer(training_df,num_cols+num_model_cols)
training_df = tf_tools.apply_numeric_imputer(training_df,num_cols+num_model_cols,num_imp_rules)

# one hot encoding - encode categorical values as 0/1
encoder_rules = tf_tools.train_encoder(training_df,cat_cols)
training_df = tf_tools.apply_encoder(training_df,encoder_rules)

# mark null targets as false
training_df = training_df.fillna(0,subset=target_cols)

# get lists of new columns (flags and encoded vars)
flag_vars = [i for i in training_df.columns if i.find("flg_") != -1]
enc_cols = [i for i in training_df.columns if i.find("enc_") != -1]

# get complete list of training columns
training_cols = num_cols + num_model_cols + flag_default_false_cols + flag_default_null_cols + enc_cols + flag_vars

# optionally subset, and export
export_df = training_df.select(master_id + training_cols + target_cols)
# export_df = export_df.sample(False,1000000/training_df.count(),seed=1928)
export_df.write.mode("overwrite").parquet(training_data_write_path)
