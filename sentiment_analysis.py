import json
import re
import chromadb
from chromadb.config import Settings
from transformers import pipeline
import os
from pathlib import Path

# === Conectar com o ChromaDB ===
def get_chroma_collection(db_dir="chroma_reviews_db"):
    db_path = Path(db_dir).resolve()

    print(f"[DEBUG] Tentando conectar ao ChromaDB em: {db_path}")
    print(f"[DEBUG] Diretório existe: {db_path.exists()}")

    # Cria o diretório se não existir (garantir que persist_directory exista)
    if not db_path.exists():
        os.makedirs(db_path)
        print(f"[DEBUG] Diretório criado manualmente em: {db_path}")
    else:
        print(f"[DEBUG] Conteúdo do diretório: {list(db_path.iterdir())}")

    # Conectar ao cliente persistente
    chroma_client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )

    # Tenta obter a coleção
    try:
        collection = chroma_client.get_collection("reviews_collection")
        print(f"[DEBUG] Coleção encontrada com {collection.count()} documentos")
        return collection
    except ValueError as e:
        print(f"[DEBUG] Coleção não encontrada: {e}")
        collection = chroma_client.create_collection("reviews_collection")
        print(f"[DEBUG] Nova coleção criada")
        return collection



# === Parsing seguro da resposta (em caso de uso futuro com LLMs) ===
def parse_llm_response(response_text):
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"[!] Erro ao parsear JSON: {e}")

    return {
        "sentiment": "Neutro",
        "summary": "Resumo não disponível",
        "positive_points": [],
        "negative_points": []
    }

# === Função principal de análise ===
def sentiment_analyzer(product_id: str) -> dict:
    print(f"[DEBUG] Analisando produto: {product_id}")

    collection = get_chroma_collection()

    # Buscar documentos específicos do produto
    results = collection.get(
        where={"product_id": product_id},
        include=["documents", "metadatas"]
    )

    print(f"[DEBUG] Documentos encontrados para product_id '{product_id}': {len(results['ids'])}")

    if not results['ids']:
        return {
            "sentiment": "Neutro",
            "summary": "Nenhum review encontrado para este produto",
            "positive_points": [],
            "negative_points": [],
            "top_reviews": [],
            "debug_info": {
                "product_id_searched": product_id,
                "total_docs_in_collection": len(collection.get()['ids']),
                "available_product_ids": list(set(m.get('product_id', 'N/A') for m in collection.get()['metadatas'] if m))
            }
        }

    reviews = results['documents']
    reviews_text = " ".join(reviews)[:4000]  # limitar tamanho

    # Gerar resumo com LLM
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        prompt_text = f"Resuma os seguintes reviews de produto, explicando se o produto foi elogiado ou criticado: {reviews_text}"
        summary_output = summarizer(prompt_text, max_length=60, min_length=20, do_sample=False)
        summary = summary_output[0]['summary_text']
    except Exception as e:
        print(f"[DEBUG] Erro na sumarização: {e}")
        summary = "Resumo não disponível devido a erro técnico"

    # Classificação de sentimento
    try:
        sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)
        sentiments = [sentiment_classifier(review)[0] for review in reviews]
    except Exception as e:
        print(f"[DEBUG] Erro na análise de sentimentos: {e}")
        sentiments = [{"label": "3", "score": 0.5} for _ in reviews]

    # Agregação
    sentiment_scores = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for s in sentiments:
        label = s['label'].upper()
        if "NEGATIVE" in label or label.startswith("1") or label.startswith("2"):
            sentiment_scores["NEGATIVE"] += 1
        elif "POSITIVE" in label or label.startswith("4") or label.startswith("5"):
            sentiment_scores["POSITIVE"] += 1
        else:
            sentiment_scores["NEUTRAL"] += 1

    if sentiment_scores["POSITIVE"] > sentiment_scores["NEGATIVE"]:
        overall_sentiment = "Positivo"
    elif sentiment_scores["NEGATIVE"] > sentiment_scores["POSITIVE"]:
        overall_sentiment = "Negativo"
    else:
        overall_sentiment = "Neutro"

    # Pontos positivos/negativos
    positive_points = [r for r, s in zip(reviews, sentiments) if s['label'].startswith("4") or s['label'].startswith("5")][:5]
    negative_points = [r for r, s in zip(reviews, sentiments) if s['label'].startswith("1") or s['label'].startswith("2")][:5]

    top_reviews = [re.sub(r'\s+', ' ', review.strip())[:200] for review in reviews[:3]]

    return {
        "sentiment": overall_sentiment,
        "summary": summary,
        "positive_points": positive_points,
        "negative_points": negative_points,
        "top_reviews": top_reviews,
        "debug_info": {
            "reviews_found": len(reviews),
            "sentiment_distribution": sentiment_scores
        }
    }

# === Teste de conexão ===
def test_connection():
    try:
        collection = get_chroma_collection()
        all_docs = collection.get()
        print(f"Conexão bem-sucedida! Total de documentos: {len(all_docs['ids'])}")

        if all_docs['metadatas']:
            product_ids = [m.get('product_id') for m in all_docs['metadatas'] if m and 'product_id' in m]
            unique_ids = sorted(set(product_ids))
            n_ids = len(unique_ids)
            sample_ids = unique_ids[:10]  # mostra só os 10 primeiros

            print(f"[DEBUG] Product IDs disponíveis: {sample_ids} (total: {n_ids})")


        return True
    except Exception as e:
        print(f"Erro na conexão: {e}")
        return False

if __name__ == "__main__":
    test_connection()