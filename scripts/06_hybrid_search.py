import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше для первинних списків
FINAL_K = 5  # скільки виводимо у фінальний топ

# Ініціалізація інструментів
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# ----------------------------------------------------------------------
# Побудова локального індексу BM25
# ----------------------------------------------------------------------
print("Будуємо локальний індекс BM25...")

# Для кращої точності об'єднуємо заголовок та анотацію
def prepare_tokens(text):
    return str(text).lower().replace("\n", " ").split()

df["bm25_text"] = df["title"] + " " + df["abstract"]
tokenized_corpus = df["bm25_text"].apply(prepare_tokens).tolist()
bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------------------------------------------------
# Реалізація окремих функцій пошуку
# ----------------------------------------------------------------------

def search_bm25(query: str, top_k: int = TOP_K):
    """ Пошук по локальному індексу BM25 """
    query_tokens = prepare_tokens(query)
    scores = bm25.get_scores(query_tokens)
    
    # Сортуємо індекси за спаданням скору
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if scores[idx] <= 0:  # Ігноруємо документи без жодного збігу слів
            continue
        results.append({
            "id": df.loc[idx, "id"],
            "title": df.loc[idx, "title"],
            "category": df.loc[idx, "category"],
            "year": df.loc[idx, "year"],
            "score": scores[idx],
            "rank": rank
        })
    return results

def search_vector(query: str, top_k: int = TOP_K):
    """ Векторний пошук у Pinecone """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res.get("matches", []), 1):
        meta = match.get("metadata", {})
        results.append({
            "id": meta.get("arxiv_id", match["id"].replace("paper_", "").replace("_", ".")),
            "title": meta.get("title", "Unknown"),
            "category": meta.get("category", "Unknown"),
            "year": meta.get("year", 0),
            "score": match["score"],
            "rank": rank
        })
    return results

# ----------------------------------------------------------------------
# Реалізація Reciprocal Rank Fusion (RRF)
# ----------------------------------------------------------------------

def run_hybrid_search(query: str, k_rrf: int = 60):
    """ Гібридний пошук через об'єднання рангів BM25 та Pinecone """
    bm25_results = search_bm25(query, top_k=TOP_K)
    vector_results = search_vector(query, top_k=TOP_K)
    
    rrf_scores = {}
    doc_metadata = {}
    
    # Обробляємо результати BM25
    for doc in bm25_results:
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + doc["rank"]))
        doc_metadata[doc_id] = doc
        
    # Обробляємо результати векторного пошуку
    for doc in vector_results:
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + doc["rank"]))
        if doc_id not in doc_metadata:
            doc_metadata[doc_id] = doc

    # Сортуємо документи за спаданням RRF скору
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    hybrid_results = []
    for doc_id, score in sorted_docs[:FINAL_K]:
        meta = doc_metadata[doc_id]
        hybrid_results.append({
            "id": doc_id,
            "title": meta["title"],
            "category": meta["category"],
            "year": meta["year"],
            "rrf_score": score
        })
    return bm25_results[:FINAL_K], vector_results[:FINAL_K], hybrid_results

# ----------------------------------------------------------------------
# Демонстрація та виведення результатів порівняння
# ----------------------------------------------------------------------

def print_results_table(title, items, is_hybrid=False):
    print(f"\n{title}")
    print("-" * 90)
    if not items:
        print("Нічого не знайдено.")
        return
    for i, item in enumerate(items, 1):
        score_str = f"RRF Score: {item['rrf_score']:.5f}" if is_hybrid else f"Score: {item['score']:.4f}"
        print(f"{i}. [{score_str}] ID: {item['id']} | Категорія: {item['category']} | Рік: {int(item['year'])}")
        print(f"   Заголовок: {item['title'][:80]}...")

def main():
    test_queries = [
        ("BERT fine-tuning", "ЗАПИТ 1 (Точний термін / абревіатура)"),
        ("Yann LeCun convolutional networks", "ЗАПИТ 2 (Ім'я автора + ключові слова)"),
        ("making computers understand human emotions from text", "ЗАПИТ 3 (Перефразування / Семантичний концепт)")
    ]
    
    for query_text, desc in test_queries:
        print("\n" + "="*90)
        print(f"{desc}: '{query_text}'")
        print("="*90)
        
        bm25_res, vec_res, hybrid_res = run_hybrid_search(query_text)
        
        print_results_table("Топ-5 за версією BM25", bm25_res)
        print_results_table("Топ-5 за версією Векторного Пошуку (Pinecone)", vec_res)
        print_results_table("Топ-5 за версією Гібридного Пошуку (RRF)", hybrid_res, is_hybrid=True)

if __name__ == "__main__":
    main()