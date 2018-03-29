import cx_Oracle
import pandas as pd
from flask import Flask, jsonify

dsn_host = ''
dsn_port = 0
dsn_service_name = ''
ora_user = ''
ora_pass = ''

sql_query = ".sql"
endpoint_str = '/api/<int:id>'
endpoint_host = ''
endpoint_port = 0


app = Flask(__name__)

eaa = cx_Oracle.makedsn(dsn_host, dsn_port, service_name=dsn_service_name)
conn = cx_Oracle.connect(ora_user,,eaa)
cur = c_sbeaa.cursor()

sql_file = open(sql_query,"r")
sql = sql_file.read()

@app.route(endpoint_str,methods=['GET'])

def get_rows(id):
  records = pd.read_sql(
    con=conn
    ,sql = sql.format(cx = id))

  return records.to_json(orient='records')

if __name__ == '__main__':
  app.run(debug=True,
    host = endpoint_host
    ,port = endpoint_port)



#this = get_rows(1087305601)

#out = this.to_json(orient="index")

#print(out)
