import os
from urllib.parse import quote_plus

user = 'user'
password = 'password'
encoded_password = quote_plus(password)
host = 'host'
port = 'port'
database = 'database'

DATABASE_URL = os.getenv('DATABASE_URL', f'mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}')