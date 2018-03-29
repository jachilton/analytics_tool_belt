import pandas as pd
import os
import paramiko
import smtplib

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.sql.functions import col, avg, udf, when, desc

import smtplib
import datetime

# Here are the email package modules we'll need
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

test_data_sql = ""
ref_data_sql =""

ssh_user = ""
ssh_pwd = ""
ssh_host = ""
drop_cols = []

email_username = ""
email_password = ""
smtp_host = ""

email_subj = ""
recipients_list = ""
smtp_sender = ""
email_preview = ""




pyspark_app_nm =  "data_tester"



app_pyspark_conf = SparkConf()
app_pyspark_conf.setAppName(pyspark_app_nm)
spark = SparkSession.builder.appName(pyspark_app_nm).enableHiveSupport().getOrCreate()


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ssh_host,username=ssh_user,password=ssh_pwd)


test_data = spark.sql(test_data_sql)
ref_data = spark.sql(ref_data_sql) #pyspark

def train_encoder(frame,fcols):

    # generate dictionary with the top 50 unique values by column

    unique_master = {}

    for icol in fcols:
    	icol_vals = frame.groupby(icol).count().orderBy("count",ascending=0).limit(50) #pyspark
    	icol_vals = icol_vals.toPandas()[icol].tolist() #pyspark
    	unique_master[icol] = icol_vals

    for icol in unique_master:
    	for val in unique_master[icol]:
    		new_field_nm = "enc_" + str(icol) + "_" + str(val).replace("+","").replace("$","").replace(" ","_").replace(".","_")
    		cols_enc.append(new_field_nm)

    return unique_master

def encodomatic(frame,unique_dict):

    # use the dictionary generated from the encoder trainer, and apply one hot encoding to create dummy vars for each value of each col

    for icol in unique_dict:
    	for val in unique_dict[icol]:
    		new_field_nm = "enc_" + str(icol) + "_" + str(val).replace("+","").replace("$","").replace(" ","_").replace(".","_")
    		frame = frame.withColumn(new_field_nm,when((frame[icol] == str(val)),1).otherwise(0)) #pyspark

    return frame


test_data_cols = set(test_data.columns) #pyspark
ref_data_cols = set(ref_data.columns) #pyspark

# get columns unique to each dataset
test_data_only_cols = list(test_data_cols - ref_data_cols)
ref_data_only_cols = list(ref_data_cols - test_data_cols)
op_cols = list(test_data_cols - (test_data_cols - ref_data_cols))

# filter columns to common columns
df_cols_types = pd.DataFrame(test_data.dtypes,columns=["name","dtype"]) #pyspark
df_cols_types = df_cols_types[df_cols_types['name'].isin(op_cols)]

# split data to numeric/string (cat)
cols_num = df_cols_types[df_cols_types["dtype"] == "double"]["name"].tolist()
cols_str = df_cols_types[df_cols_types["dtype"] == "string"]["name"].tolist()

# apply final column drop cleanup to ensure clean lists
cols_num = list(set(cols_num) - set(drop_cols) - set(test_data_only_cols) - set(ref_data_only_cols))
cols_str = list(set(cols_str) - set(drop_cols) - set(test_data_only_cols) - set(ref_data_only_cols))


# train the encoder (generate uniques dict)
cols_enc = []
unique_master = train_encoder(ref_data,cols_str) #pyspark

# apply the encoder
test_data_enc = encodomatic(test_data,unique_master) #pyspark
ref_data_enc = encodomatic(ref_data,unique_master) #pyspark

analysis_cols = cols_num + cols_enc


#test_data_enc.limit(10).show()


sftp = ssh.open_sftp()

test_data_stats = test_data_enc.select(analysis_cols).describe().toPandas().T
ref_data_stats = ref_data_enc.select(analysis_cols).describe().toPandas().T


table_html = open("table_html.html","w")

results_joined = ref_data_stats.join(test_data_stats,lsuffix="ctl_",rsuffix="test_")
results_joined.to_html(
    buf=table_html
    ,header=True
    )

table_html.close()


sftp = ssh.open_sftp()





COMMASPACE = ', '

msg = MIMEMultipart("Test")
msg['Subject'] = email_subj
recipients = recipients_list
msg['From'] = smtp_sender
msg['To'] = COMMASPACE.join(recipients)
msg.preamble = email_preview


html_body = """

<html>
<head></head>

<body>

<p>
<br>
<br>
</p>
</html>

"""
table_html = open("table_html.html","r")
attach_body = MIMEText(html_body.format(result_table=table_html.read()),"html")
msg.attach(attach_body)
table_html.close()

my_file1 = "test_data_stats.csv"

fp1 = open(my_file1,'rb')
attachment1 = MIMEBase("text","csv")
attachment1.set_payload(fp1.read())
attachment1.add_header("Content-Disposition", "attachment", filename=my_file1)
fp1.close()
encoders.encode_base64(attachment1)
msg.attach(attachment1)

my_file2 = "ref_data_stats.csv"

fp2 = open(my_file2,'rb')
attachment2 = MIMEBase("text","csv")
attachment2.set_payload(fp2.read())
attachment2.add_header("Content-Disposition", "attachment", filename=my_file2)
fp2.close()
encoders.encode_base64(attachment2)
msg.attach(attachment2)

# Send the email via our own SMTP server.
server = smtplib.SMTP(smtp_host)
server.ehlo()
server.starttls()
server.login(email_username,email_password)
server.send_message(msg)
server.quit()
