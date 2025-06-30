import pandas as pd
import re
import os
import numpy as np
from transformers import pipeline
import json
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from uuid import uuid4

df = pd.read_csv('processed_text.csv')

# Inicializar o modelo
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Gerar embeddings
embeddings = model.encode(df['review_comment_message'].tolist(), show_progress_bar=True)

# Garantir que product_id seja str
df['product_id'] = df['product_id'].astype(str)


# Criar o diretório do banco vetorial, se não existir
db_path = "chroma_reviews_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)
    print(f"📁 Diretório criado: {db_path}")
else:
    print(f"📁 Diretório já existia: {db_path}")


# 1. Inicializa o ChromaDB em modo persistente
chroma_client = chromadb.PersistentClient(
    path="chroma_reviews_db",
    settings=Settings(anonymized_telemetry=False)
)


# Apaga a coleção antiga (caso exista) para garantir dados atualizados
try:
    chroma_client.delete_collection("reviews_collection")
except:
    pass  # ignora se não existir

# 2. Cria coleção nova
collection = chroma_client.create_collection(
    name="reviews_collection"
)

# 3. Preparar dados para inserção

# IDs únicos para cada vetor (Chroma exige isso)
ids = [str(uuid4()) for _ in range(len(df))]

# Textos (reviews limpos)
documents = df['review_comment_message'].tolist()

# Metadados: product_id e review_score
metadatas = [
    {
        "product_id": pid,
        "review_score": int(score)
    }
    for pid, score in zip(df['product_id'], df['review_score'])
]

# Inserir na coleção com embeddings já calculadas
BATCH_SIZE = 5000

n = len(df)
for i in range(0, n, BATCH_SIZE):
    print(f"Inserindo batch {i} a {min(i + BATCH_SIZE, n)}...")
    
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE].tolist()
    batch_metadatas = metadatas[i:i+BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

# Verificar se o banco foi criado
if os.path.exists("chroma_reviews_db"):
    print("O banco vetorial foi criado.")
else:
    print("O banco vetorial NÃO existe.")


# Reabrir o cliente (simulando nova sessão)
client = chromadb.PersistentClient(
    path="chroma_reviews_db",
    settings=Settings(anonymized_telemetry=False)
)

# Verificar coleções existentes
collections = client.list_collections()
print("Coleções existentes:", [c.name for c in collections])

# Carrega a coleção (caso já esteja criada)
collection = client.get_collection("reviews_collection")

# Quantidade de embeddings armazenados
n_docs = collection.count()
print(f"🔍 Total de documentos na coleção: {n_docs}")


