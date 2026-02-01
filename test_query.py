# test_query.py
import requests
import json

def test_backend_connection():
    """Test if backend is accessible"""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_query_endpoint():
    """Test the query endpoint"""
    query = "What is the name and qualification mentioned in the resume documents?"
    
    try:
        response = requests.post(
            "http://127.0.0.1:5000/query",
            json={"query": query, "use_agentic": True},
            timeout=30  # Longer timeout for query processing
        )
        
        print(f"\n✅ Query endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"\n❌ Query endpoint failed: {e}")
        return False

def test_quick_query():
    """Test quick query endpoint (simpler)"""
    query = "What names are in the documents?"
    
    try:
        response = requests.post(
            "http://127.0.0.1:5000/quick_query",
            json={"query": query},
            timeout=30
        )
        
        print(f"\n✅ Quick query endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
        return True
    except Exception as e:
        print(f"\n❌ Quick query failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Backend Connection...")
    test_backend_connection()
    
    print("\n🔍 Testing Query Endpoint...")
    test_query_endpoint()
    
    print("\n🔍 Testing Quick Query Endpoint...")
    test_quick_query()