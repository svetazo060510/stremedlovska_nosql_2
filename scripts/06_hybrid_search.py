import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings

# Вимикаємо зайві попередження бібліотек для чистого виводу в консоль
warnings.filterwarnings("ignore")
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # Ширша вибірка кандидатів
FINAL_K = 5  # Кількість документів, що виводяться у фінальний гібридний топ

# Ініціалізація інструментів та перевірка конфігурації
if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("PINECONE_API_KEY не знайдено у файлі .env!")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

# reset_index(drop=True) для точного мапування локальних рядків
# із системними ідентифікаторами Pinecone (формату paper_0 ... paper_9999)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# ----------------------------------------------------------------------
# Побудова локального індексу BM25
# ----------------------------------------------------------------------
print("Будуємо локальний індекс BM25...")

def prepare_tokens(text):
    """ Оптимізована токенізація з очищенням від базової пунктуації """
    clean_text = str(text).lower().replace("\n", " ")
    # Видаляємо знаки пунктуації, щоб зберегти чисті токени й абревіатури
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    return clean_text.split()

# Об'єднуємо заголовок та анотацію для кращого лексичного охоплення алгоритмом BM25
df["bm25_text"] = df["title"] + " " + df["abstract"]
tokenized_corpus = df["bm25_text"].apply(prepare_tokens).tolist()
bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------------------------------------------------
# Реалізація окремих функцій пошуку
# ----------------------------------------------------------------------

def search_bm25(query: str, top_k: int = TOP_K):
    """ Лексичний пошук по локальному індексу BM25 """
    query_tokens = prepare_tokens(query)
    scores = bm25.get_scores(query_tokens)
    
    # Сортуємо індекси документів за спаданням скору
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if scores[idx] <= 0:  # Ігноруємо документи, де немає жодного збігу слів із запитом
            continue
        results.append({
            "id": str(df.loc[idx, "id"]),  # Оригінальний чистий arXiv ID (наприклад, "0903.1967")
            "title": df.loc[idx, "title"],
            "category": df.loc[idx, "category"],
            "year": df.loc[idx, "year"],
            "score": scores[idx],
            "rank": rank
        })
    return results

def search_vector(query: str, top_k: int = TOP_K):
    """ Векторний пошук у Pinecone із синхронізацією ID через індекси та метадані """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res.get("matches", []), 1):
        # Спроба швидкого мапування через фізичний індекс локального DataFrame
        try:
            row_idx = int(match["id"].replace("paper_", ""))
            arxiv_id = str(df.loc[row_idx, "id"])
            title = df.loc[row_idx, "title"]
            category = df.loc[row_idx, "category"]
            year = df.loc[row_idx, "year"]
        except (ValueError, KeyError):
            # Якщо пряме мапування рядків збіглося, дістаємо дані строго з метаданих хмари
            meta = match.get("metadata", {})
            title = meta.get("title", "Unknown")
            category = meta.get("category", "Unknown")
            year = meta.get("year", 0)
            
            # Перевіряємо наявність оригінального чистий ID в метаданих
            if "arxiv_id" in meta:
                arxiv_id = str(meta["arxiv_id"])
            else:
                # Резервна деструктуризація технічного ID з примусовим відновленням крапки
                # (наприклад, chunk_fixed_0903_1967_0 -> 0903_1967 -> 0903.1967)
                raw_id = match["id"]
                clean_id = raw_id.replace("chunk_fixed_", "").replace("chunk_semantic_", "").replace("paper_", "")
                clean_id = re.sub(r'_\d+$', '', clean_id)  # Відсікаємо хвіст номера чанка, якщо він є
                arxiv_id = clean_id.replace("_", ".")      # Повертаємо крапку для повного збігу з BM25 та df["id"]

        results.append({
            "id": arxiv_id,  # Гарантовано збігається з чистим форматом ID у локальному BM25!
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
    
    # Акумулюємо RRF-бали для результатів локального лексичного пошуку BM25
    for doc in bm25_results:
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + doc["rank"]))
        doc_metadata[doc_id] = doc
        
    # Акумулюємо RRF-бали для результатів хмарного векторного пошуку Pinecone
    for doc in vector_results:
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + doc["rank"]))
        if doc_id not in doc_metadata:
            doc_metadata[doc_id] = doc

    # Сортуємо отриманий словник документів за спаданням фінального RRF-скору
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
        print("Нічого не знайдено за цим запитом.")
        return
    for i, item in enumerate(items, 1):
        score_str = f"RRF Score: {item['rrf_score']:.5f}" if is_hybrid else f"Score: {item['score']:.4f}"
        print(f"{i}. [{score_str}] ID: {item['id']} | Категорія: {item['category']} | Рік: {int(item['year'])}")
        print(f"   Заголовок: {item['title'][:85]}...")

def main():
    # Набір тестових запитів для валідації системи
    test_queries = [
        ("BERT fine-tuning", "ЗАПИТ 1 (Точний термін / абревіатура)"),
        ("Yann LeCun convolutional networks", "ЗАПИТ 2 (Ім'я автора + ключові слова)"),
        ("making computers understand human emotions from text", "ЗАПИТ 3 (Перефразування / Семантичний концепт)")
    ]
    
    for query_text, desc in test_queries:
        print("\n" + "="*90)
        print(f"{desc}: '{query_text}'")
        print("="*90)
        
        # Виконуємо повне гібридне ранжування
        bm25_res, vec_res, hybrid_res = run_hybrid_search(query_text)
        
        # Виводимо порівняльні блоки результатів для README.md
        print_results_table("Топ-5 за версією BM25", bm25_res)
        print_results_table("Топ-5 за версією Векторного Пошуку (Pinecone)", vec_res)
        print_results_table("Топ-5 за версією Гібридного Пошуку (RRF)", hybrid_res, is_hybrid=True)

if __name__ == "__main__":
    main()