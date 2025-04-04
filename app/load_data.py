import redis
from sentence_transformers import SentenceTransformer

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# Obtener todas las claves con KEYS
all_keys = r.keys("doc:*")

# Contar claves base únicas
document_keys = set()
for key in all_keys:
    base_key = key.split(":")[0] + ":" + key.split(":")[1]
    document_keys.add(base_key)

# Número total de documentos
num_docs = len(document_keys) + 1

model = SentenceTransformer('all-mpnet-base-v2')  

documents = [] 

paragraph = input("Ingrese un párrafo de texto:\n")  
documents.append(paragraph)

doc = "doc:" + str(num_docs)
document_keys = [doc]

embeddings = model.encode(documents).tolist()
for key, doc, emb in zip(document_keys, documents, embeddings):
    r.set(key + ":text", doc)
    r.set(key + ":emb", ','.join(map(str, emb)))

print("Datos de documentos almacenados en Redis")