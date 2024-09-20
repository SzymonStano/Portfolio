import os
from urllib.parse import quote_plus

user = 'team08'
password = 'te@m24ob'
encoded_password = quote_plus(password)
host = 'giniewicz.it'
port = '3306'
database = 'team08'

DATABASE_URL = os.getenv('DATABASE_URL', f'mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}')