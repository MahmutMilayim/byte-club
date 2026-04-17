import chromadb
from chromadb.config import Settings
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")

print("--- RAG SYSTEM VERIFICATION ---")

# 1. Check DB Path
db_path = "./chroma_db"
if not os.path.exists(db_path):
    print(f"ERROR: Database path {db_path} does not exist.")
    exit(1)
print(f"✓ Database path found: {db_path}")

try:
    # 2. Connect to ChromaDB
    # Using persistent client in read mode (default)
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    print("✓ Connected to ChromaDB")
    
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not found in environment. Please set it in .env")
        exit(1)
    
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY, 
        model_name="models/gemini-embedding-001"
    )

    # 3. Get Collection
    try:
        collection = client.get_collection("football_rules_google", embedding_function=google_ef)
        print("✓ Collection 'football_rules_google' found")
    except Exception as e:
        print(f"ERROR: Collection not found: {e}")
        exit(1)

    # 4. Check Content
    count = collection.count()
    print(f"✓ Total Documents Indexed: {count}")
    
    if count == 0:
        print("WARNING: Collection is empty!")
    else:
        # Check first document
        first_doc = collection.get(limit=1)
        print(f"  - Sample ID: {first_doc['ids'][0]}")
        print(f"  - Sample Metadata: {first_doc['metadatas'][0]}")
        # print(f"  - Sample Content (truncated): {first_doc['documents'][0][:100]}...")

    # 5. Test Vector Search
    queries = [
        "kırmızı kart gerektiren fauller nelerdir",
        "ofsayt kuralı nedir",
        "elle oynama ne zaman ihlaldir"
    ]
    
    print("\n--- VECTOR SEARCH TEST ---")
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = collection.query(
            query_texts=[q],
            n_results=1
        )
        
        if results['documents'][0]:
            print(f"  ✓ Result Found (Distance: {results['distances'][0][0]:.4f})")
            print(f"  ✓ Source: {results['metadatas'][0][0].get('name', 'Unknown')}")
            print(f"  ✓ Content:\n{results['documents'][0][0]}")
            print("-" * 40)
        else:
            print("  FAIL: No results found")

except Exception as e:
    print(f"\nERROR during verification: {e}")
    print("NOTE: Verify that backend.py is not locking the database if using SQLite.")
