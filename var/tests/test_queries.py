import chromadb
from chromadb.config import Settings
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY', '')

db_path = './chroma_db'
client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY, 
    model_name='models/gemini-embedding-001'
)

collection = client.get_collection('football_rules_google', embedding_function=google_ef)

queries = [
    'kırmızı kart gerektiren fauller nelerdir',
    'ofsayt kuralı nedir',
    'elle oynama ne zaman ihlaldir',
    'kaleci ceza sahası dışında topa elle dokunursa ne olur',
    'top çizgiyi geçmeden gol olur mu',
    'sarı kart üstüne sarı kart',
    'penaltı atışı sırasında kaleci ihlali',
    'ofsayt olmayan pozisyonlar neler'
]

for q in queries:
    print(f"\nQuery: {q}")
    results = collection.query(query_texts=[q], n_results=1)
    if results['documents'] and results['documents'][0]:
        print(f"Distance: {results['distances'][0][0]:.4f}")
        print(f"Source: {results['metadatas'][0][0].get('name', 'Unknown')}")
        content = results['documents'][0][0].replace('\n', ' ')
        print(f"Content: {content[:150]}...")
    else:
        print("No results found.")
