import os
import re
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
TOP_K = 10   # Беремо ширше, щоб RRF міг переранжувати
FINAL_K = 5  # Скільки виводимо у фінальний топ

# Ініціалізація інструментів
if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("PINECONE_API_KEY не знайдено у файлі .env!")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

# reset_index(drop=True) критично важливий для точного мапування embeddings та Pinecone IDs (paper_0..paper_9999)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# ----------------------------------------------------------------------
# Побудова локального індексу BM25
# ----------------------------------------------------------------------
print("Будуємо локальний індекс BM25...")

def prepare_tokens(text):
    """ Токенізація з очищенням від базової пунктуації """
    clean_text = str(text).lower().replace("\n", " ")
    # Видаляємо пунктуацію, залишаючи слова та цифри
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    return clean_text.split()

# Об'єднуємо заголовок та анотацію для кращого лексичного охоплення
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
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if scores[idx] <= 0:  # Ігноруємо документи без жодного лексичного збігу
            continue
        results.append({
            "id": str(df.loc[idx, "id"]),  # Оригінальний arXiv ID (наприклад, "1401.3753")
            "title": df.loc[idx, "title"],
            "category": df.loc[idx, "category"],
            "year": df.loc[idx, "year"],
            "score": scores[idx],
            "rank": rank
        })
    return results

def search_vector(query: str, top_k: int = TOP_K):
    """ Векторний пошук у Pinecone із синхронізацією ID через індекси """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res.get("matches", []), 1):
        # Дістаємо числовий індекс із ID типу "paper_142"
        try:
            row_idx = int(match["id"].replace("paper_", ""))
            # Беремо метадані з локального DataFrame за цим індексом
            arxiv_id = str(df.loc[row_idx, "id"])
            title = df.loc[row_idx, "title"]
            category = df.loc[row_idx, "category"]
            year = df.loc[row_idx, "year"]
        except (ValueError, KeyError):
            # Якщо структура ID у хмарі відрізняється
            meta = match.get("metadata", {})
            arxiv_id = meta.get("arxiv_id", match["id"])
            title = meta.get("title", "Unknown")
            category = meta.get("category", "Unknown")
            year = meta.get("year", 0)

        results.append({
            "id": arxiv_id, 
            "title": title,
            "category": category,
            "year": year,
            "score": match["score"],
            "rank": rank
        })
    return results

# ----------------------------------------------------------------------
# Реалізація Reciprocal Rank Fusion (RRF)
# ----------------------------------------------------------------------

def run_hybrid_search(query: str, k_rrf: int = 60):
    """ Гібридний пошук через об'єднання рангів BM25 та Pinecone за алгоритмом RRF """
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
# Виведення результатів порівняння
# ----------------------------------------------------------------------

def print_results_table(title, items, is_hybrid=False):
    print(f"\n{title}")
    print("-" * 90)
    if not items:
        print("Нічого не знайдено за цим запитом.")
        return
    for i, item in enumerate(items, 1):
        score_str = f"RRF Score: {item['rrf_score']:.5f}" if is_hybrid else f"Score: {item['score']:.4f}"
        print(f"{i}. [{score_str}] ID: {item['id']} | Категорія: {item['category']} | Рік: {int(item['year'])}")
        print(f"   Заголовок: {item['title'][:85]}...")

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