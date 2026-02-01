# test_milvus_config.py
import os
from dotenv import load_dotenv
from pymilvus import connections, utility

load_dotenv()

print("🔧 Checking Milvus Configuration...")
print("="*50)

# Check required environment variables
required_vars = [
    'MILVUS_HOST',
    'MILVUS_PORT', 
    'MILVUS_COLLECTION_NAME',
    'EMBEDDING_DIMENSION'
]

all_present = True
for var in required_vars:
    value = os.getenv(var)
    status = "✅" if value else "❌"
    print(f"{status} {var}: {value}")
    if not value:
        all_present = False

print("\n🔌 Testing Milvus Connection...")
try:
    connections.connect(
        host=os.getenv('MILVUS_HOST', 'localhost'),
        port=int(os.getenv('MILVUS_PORT', '19530'))
    )
    print("✅ Successfully connected to Milvus!")
    
    # Show current collections
    collections = utility.list_collections()
    print(f"📚 Collections: {collections}")
    
    # Check if our collection exists
    target_collection = os.getenv('MILVUS_COLLECTION_NAME')
    if target_collection in collections:
        print(f"✅ Collection '{target_collection}' exists")
    else:
        print(f"⚠️  Collection '{target_collection}' doesn't exist (will be created on first use)")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")

print("="*50)


## Using Docker (easiest)
"docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.3"