from pymilvus import connections, utility

try:
    # Connect to Milvus
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    
    print("✅ Successfully connected to Milvus!")
    
    # Check server version
    version = utility.get_server_version()
    print(f"📦 Milvus version: {version}")
    
    # List collections (should be empty initially)
    collections = utility.list_collections()
    print(f"📚 Collections: {collections}")
    
    # Check connection status
    print(f"🔌 Connection status: {connections.has_connection('default')}")
    
except Exception as e:
    print(f"❌ Error: {e}")