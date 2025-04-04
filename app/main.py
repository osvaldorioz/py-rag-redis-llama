from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import rag_module 
import json

app = FastAPI()
rag = rag_module.RAG("127.0.0.1", 6379)

@app.post("/rag-redis-llama-query")
def rag_paradigm(pregunta: str):
    
    query = pregunta
    document_keys = rag.get_document_keys()
    response = rag.generate_response(query, document_keys)
    
    j1 = {
        "pregunta": pregunta, 
        "respuesta": response
    }
    jj = json.dumps(str(j1))

    return jj
"""
@app.post("/rag-redis-llama-load-document")
def load_document(documento: str):
    
    document_keys = rag.get_document_keys()
    num_docs = len(document_keys) + 1

    model = SentenceTransformer('all-mpnet-base-v2')  

    documents = [] 

    documents.append(documento)

    doc = "doc:" + str(num_docs)
    document_keys = [doc]

    embeddings = model.encode(documents).tolist()
    for key, doc, emb in zip(document_keys, documents, embeddings):
        r.set(key + ":text", doc)
        r.set(key + ":emb", ','.join(map(str, emb)))

    print("Datos de documentos almacenados en Redis")
    j1 = {
        "documento": doc, 
        "respuesta": response
    }
    jj = json.dumps(str(j1))

    return jj
"""