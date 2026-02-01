# clear_milvus.py
from pymilvus import connections, utility
from backend.config import config

# Connect to Milvus
connections.connect(
    alias="default",
    host=config.MILVUS_HOST,
    port=config.MILVUS_PORT,
    user=config.MILVUS_USER if config.MILVUS_USER else None,
    password=config.MILVUS_PASSWORD if config.MILVUS_PASSWORD else None,
    db_name=config.MILVUS_DB_NAME
)

# Check and drop collection if exists
collection_name = config.MILVUS_COLLECTION_NAME
if utility.has_collection(collection_name):
    print(f"Dropping collection: {collection_name}")
    utility.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' dropped successfully!")
else:
    print(f"⚠️ Collection '{collection_name}' doesn't exist")

# List all collections to verify
collections = utility.list_collections()
print(f"📚 Available collections: {collections}")