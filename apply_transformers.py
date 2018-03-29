import json
import smtplib
import os
import pandas as pd
import numpy as np
import pyspark


import tf_tools # ( •_•)>⌐■-■ (⌐■_■)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, when
from pyspark.sql.types import StructType

pyspark_app_nm = "pyspark_testing"
trainer_df_prqt_path = "/home/jachilto/car_sample_20180219"

cat_imputer_rules_path = "cat_imputer_rules.json"
encoder_rules_path = "encoder_rules.json"
num_imputer_rules_path = "num_imputer_rules.json"
scaler_rules_path = "scaler_rules.json"


null_fields_list = []

spark = SparkSession.builder.appName(pyspark_app_nm).getOrCreate()

print(spark.sparkContext.getConf().getAll())

trainer_df = spark.read.parquet(trainer_df_prqt_path)

cat_imputer_rules = json.load(open(cat_imputer_rules_path,'r'))
encoder_rules = json.load(open(encoder_rules_path,'r'))
num_imputer_rules = json.load(open(num_imputer_rules_path,'r'))
scaler_rules = json.load(open(scaler_rules_path,'r'))


cols_num, cols_str,col_types = tf_tools.sort_cols(trainer_df)
cols_num = list(set(cols_num) - set(null_fields_list))


trainer_df = tf_tools.apply_categorical_imputer(trainer_df,cols_str,cat_imputer_rules,True)


df_json_vals = trainer_df.toJSON().collect()[0]
df_json_schema = trainer_df.schema.jsonValue()

full_scoring_record = {"vals":json.loads(df_json_vals),"schema":df_json_schema}
json.dump({"vals":json.loads(df_json_vals),"schema":df_json_schema},"full_scoring_record.json")

spark.createDataFrame(full_scoring_record['vals'])

json.dumps({"vals":json.loads(df_json_vals),"schema":df_json_schema})


with open('json_vals.json','w') as json_vals_out_file:
    json_vals_out_file.write(df_json_vals)

with open('json_schema.json','w') as json_schema_out_file:
    json.dump(df_json_schema,json_schema_out_file)


with open('json_vals.json','r') as json_vals_in_file:
    in_df_json_vals = json.load(json_vals_in_file)

with open('json_schema.json','r') as json_schema_in_file:
    in_df_json_schema = StructType.fromJson(json.load(json_schema_in_file))


in_df = spark.read.json(
    'json_vals.json'
    ,schema = in_df_json_schema)
