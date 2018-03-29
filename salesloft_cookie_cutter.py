
import pandas as pd
import numpy as np
import psycopg2
import datetime
import random
import time
from pandas.tseries import offsets

from sqlalchemy import create_engine
from pandas.io.sql import get_schema

user = ''
password = ''
host = ''

# psycopg2
try:
    conn = psycopg2.connect("dbname='minerva' host='{host}' port='5432' user='{user}' password='{password}'".format(host = host, user=user, password=password))
    cur = conn.cursor()
    conn.autocommit = True
except Exception as e: 
    print("I am unable to connect to the database")
    print(e)


sql_query = """SELECT {values} FROM {table}"""

df = pd.read_sql(con=conn,sql=sql_query)
