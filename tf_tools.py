import json
import smtplib
import os
import pandas as pd
import numpy as np

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, when

def sort_cols(input_df):

    # Create a sorting table with columns by dtype
    input_df_col_types = pd.DataFrame(input_df.dtypes,columns=["name","dtype"])

    # Create lists of numeric/string typed cols
    cols_str = input_df_col_types[input_df_col_types["dtype"] == "string"]["name"].tolist()
    cols_num = input_df_col_types[input_df_col_types["dtype"] == "double"]["name"].tolist() + \
        input_df_col_types[input_df_col_types["dtype"] == "int"]["name"].tolist() + \
        input_df_col_types[input_df_col_types["dtype"] == "bigint"]["name"].tolist()

    cols_str.remove("node")

    cols_sort = {"cols_str":cols_str
        ,"cols_num":cols_num}

    with open("cols_sorted.json","w") as f:
        json.dump(cols_sort,f)
        f.close()

    return cols_num, cols_str,input_df_col_types

def train_categorical_imputer(frame,fcols):

    # generate dictionary with the top value by column

    modes_master = {}

    for icol in fcols:
        icol_val = frame.groupby(icol).count().orderBy("count",ascending=0).limit(1)
        icol_val = str(icol_val.toPandas()[icol].tolist()[0]).replace("+","").replace("$","").replace(" ","_").replace(".","_")
        modes_master[icol] = icol_val

    with open("cat_imputer_rules.json","w") as f:
        json.dump(modes_master,f)
        f.close()

    return modes_master

def apply_categorical_imputer(frame,fcols,imputer_values,isBulk):

    imputed_df = frame.fillna(
        value = imputer_values
        ,subset=fcols)

    df_schema = imputed_df.schema

    if isBulk == True:
        return imputed_df
    else:
        return jsonify({"payload":imputed_df.toPandas().to_dict(orient='columns'),"df_schema":df_schema})


def train_numeric_imputer(frame,fcols):

    means_master = {}

    exprs = {x: "mean" for x in fcols}
    means = frame.groupBy().agg(exprs)
    means = means.toPandas()

    clean_col_names = [str(i)[4:-1]  for i in means.columns.tolist()]
    replacement_col_mapper = dict(zip(means.columns.tolist(),clean_col_names))
    means.rename(columns=replacement_col_mapper,inplace=True)
    imputer_values = means.to_dict(orient="records")[0]

    with open("num_imputer_rules.json","w") as f:
        json.dump(imputer_values,f)
        f.close()

    return imputer_values

def apply_numeric_imputer(frame,fcols,imputer_values):

    imputed_df = frame.fillna(value = imputer_values,subset=fcols)
    df_schema = imputed_df.schema

    return imputed_df

def train_encoder(frame,fcols):

    # generate dictionary with the top 50 unique values by column

    encoder_vals = []
    unique_master = {}

    for icol in fcols:
    	icol_vals = frame.groupby(icol).count().orderBy("count",ascending=0).limit(50) #pyspark
    	icol_vals = icol_vals.toPandas()[icol].tolist() #pyspark
    	unique_master[icol] = icol_vals

    for icol in unique_master:
    	for val in unique_master[icol]:
    		new_field_nm = "enc_" + str(icol) + "_" + str(val).replace("+","").replace("$","").replace(" ","_").replace(".","_")
    		encoder_vals.append(new_field_nm)

    with open("encoder_rules.json","w") as f:
        json.dump(unique_master,f)
        f.close()

    return unique_master


def apply_encoder(frame,encoder_vals):

    # use the dictionary generated from the encoder trainer, and apply one hot encoding to create dummy vars for each value of each col

    for icol in encoder_vals:
    	for val in encoder_vals[icol]:
    		new_field_nm = "enc_" + str(icol) + "_" + str(val).replace("+","").replace("$","").replace(" ","_").replace(".","_").replace("-","_")
    		frame = frame.withColumn(new_field_nm,when((frame[icol] == str(val)),1).otherwise(0)) #

    df_schema = frame.schema

    return frame

def train_scaler(frame,fcols):


    means_master = {}

    mu_exprs = {x: "mean" for x in fcols}
    mu = frame.groupBy().agg(mu_exprs)
    mu = mu.toPandas()
    mu_clean_col_names = [str(i)[4:-1]  for i in mu.columns.tolist()]
    mu_replacement_col_mapper = dict(zip(mu.columns.tolist(),mu_clean_col_names))
    mu.rename(columns=mu_replacement_col_mapper,inplace=True)
    mu_values = mu.T.to_dict(orient="dict")[0]

    sigma_exprs = {x: "stddev_samp" for x in fcols}
    sigma = frame.groupBy().agg(sigma_exprs)
    sigma = sigma.toPandas()
    sigma_clean_col_names = [str(i)[12:-1] for i in sigma.columns.tolist()]
    sigma_replacement_col_mapper = dict(zip(sigma.columns.tolist(),sigma_clean_col_names))
    sigma.rename(columns=sigma_replacement_col_mapper,inplace=True)
    sigma_values = sigma.T.to_dict(orient="dict")[0]

    scaler_values = {"mu": mu_values, "sigma": sigma_values}

    with open("scaler_rules.json","w") as f:
        json.dump(scaler_values,f)
        f.close()


    return scaler_values



def apply_scaler(frame,fcols,scaler_values):

    for i in fcols:
        scl_mean = scaler_values['mu'][i]
        scl_stddev = scaler_values['sigma'][i]
        scaled_field_name = "scl_" + i
        frame = frame.withColumn(scaled_field_name,((frame[i] - scl_mean) / scl_stddev))

    df_schema = frame.schema

    return frame
