import os 
INIT_INDEX = os.getenv('Init_index', 'false').lower() == 'true'
# dictory for vector embeddings
INDEX_PERSIST_DICTORY = os.getenv('INDEX_PERSIST_DICTORY', './data/chromadb')

#target URL
TARGET_URL = os.getenv('TARGET_URL', 'https://www.winccoa.com/documentation/WinCCOA/latest/en_US/')

# http port ti access
HTTP_PORT = os.getenv('HTTP_PORT', 7645)

# mongodb config host, username and password
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
MONGO_PORT = os.getenv('MONGO_PORT', 27017)
MONGO_USER = os.getenv('MONGO_USER', 'testuser')
MONGO_PASS = os.getenv('MONGO_PASS', 'testpass')
