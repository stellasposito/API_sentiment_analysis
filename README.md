# 😃 Analisador de Sentimentos 😠

## 📋 Descrição do Projeto

Este projeto implementa um analisador de sentimentos para reviews de produtos utilizando técnicas de RAG (Retrieval-Augmented Generation). A solução processa reviews do dataset Olist (disponível via kaggle), cria embeddings vetoriais e oferece uma API para análise de sentimentos baseada em product_id com as respostas geradas por LLM.

## 🏗️ Arquitetura do Sistema

```
📁 API_sentiment_analysis/
├── embeddings.py
├── exploratory.ipynb
├── main.py
├── requirements.txt
├── sentiment_analysis.py

```

## 📄 Datasets utilizados
- olist_order_reviews_dataset.csv
- olist_order_items_dataset.csv

## Ordem para execução:
1. Rodar jupyter notebook com os dois arquivos para geração do csv processado.
2. Gerar embeddings com embeddings.py (esta etapa precisa da geração do csv processado).
3. Gerar repostas do LLM com sentiment_analysis.py.
4. Criar API com main.py.
5. Testar API. 

## 🚀 Etapa 1: Análise Exploratória e RAG

### 📄 `exploratory.ipynb`
**Objetivo Principal:** Importante para conseguir relacionar uma review com um product_id e realizar testes antes de rodar scripts
- Nesta etapa, fiz uma análise individual de cada dataset para verificar colunas que poderiam trazer informações importantes, e analisar qual chave única usaria para relacionar ambas planilhas.
- Esta análise inicial foi importante para entender o contexto dos dados e verificar qual a melhor forma de unir as duas tabelas sem perder informações valiosas.
- Este arquivo está dividido em:
  - *Análise Exploratória:* verificação do número de linhas, colunas, valores nulos, duplicados, notas mais frequentes e análises estatísticas de variáveis.
  - *Processamento do texto:* com técnicas de NLP, algumas pontuações foram removidas assim como caracteres especiais que poderiam prejudicar o entendimento do modelo.
  - *Salvamento de arquivo processado:* nesta etapa, o arquivo que será utilizado para geração de embeddings e análise de sentimentos é salvo.
  - *Testes de embeddings:* aqui, usei esse espaço para testar se as embeddings foram criadas corretamente.
  - *Análise de sentimentos:* antes de utilizar LLM para analisar e gerar resumos dos reviews, criei uma regra de análise de sentimentos com base no review e na nota, apenas para teste de classificação de sentimento pelo modelo nlptown/bert-base-multilingual-uncased-sentiment, e testei uma busca vetorial por similaridade.
  - *Testes LLM:* aqui, vários modelos disponíveis no Hugghing Face foram testados, até encontrar o que melhor se adaptou aos dados. Por utilizar versões gratuitas e por questões de memória do meu CPU, precisei dividir em duas tarefas de dois modelos diferentes - um para resumir e outro para classificar os reviews.

### 📄 `embeddings.py`
**Objetivo Principal:** Geração de embeddings para os textos e armazenamento no banco vetorial utilizando ChromaDB
- Utilizei um modelo de embedding pré-treinado (paraphrase-multilingual-MiniLM-L12-v2), que performa bem com a lingua portuguesa.
- Processei os textos dos reviews em lotes para otimizar performance.
- Salvei os embeddings junto com metadados (product_id, review_id, score) no banco vetorial criado.

## 🧠 Etapa 2: Análise de Sentimentos com LLM

### 📄 `sentiment_analysis.py`
**Objetivo Principal: Gerar análise de sentimentos das reviews relacionadas a cada produto** 
- Aqui, o banco vetorial é chamado e nele é realizada a busca pelas reviews.
- Utilizei o modelo "facebook/bart-large-cnn" para geração dos resumos e "nlptown/bert-base-multilingual-uncased-sentiment" para classificação do sentimento (o qual já havia sido testado anteriormente).
- Para o resumo, fiz um prompt para o modelo entender o melhor formato esperado.
- Para a classificação, os sentimentos foram classificados como Positivo, Negativo ou Neutro.
- Além disso, também foi gerado tópicos positivos e negativos do produto e os top reviews.

## 🌐 Etapa 3: Exposição via API

### 📄 `main.py`
**Objetivo:** Criação de um servidor web com FastAPI
- A API implementa endpoint `/analyze_sentiment`.
- Valida entrada (product_id deve ser string não vazia).
- Integra com o sentiment_analyzer para processamento.
- Implementa tratamento de erros e respostas padronizadas.
- Adiciona logging para monitoramento e debugging.
- Inclui documentação automática Swagger/OpenAPI.
- Implementa rate limiting para evitar sobrecarga.
- Retorna resposta JSON estruturada com a seguinte estrutura:

```json
{
  "product_id_1": [
    {
      "sentiment": "Neutro",
      "summary": "Nenhum review encontrado para este produto",
      "positive_points": [],
      "negative_points": [],
      "top_reviews": [],
    }
  ]
}
```

