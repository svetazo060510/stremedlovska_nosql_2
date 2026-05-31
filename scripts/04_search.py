import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore") # Ігнорувати всі системні попередження

# Завантажуємо змінні оточення
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# Ініціалізація інструментів
if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("PINECONE_API_KEY не знайдено у файлі .env!")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

def encode_query(query: str) -> list:
    """ Функція кодування текстового запиту в нормалізований ембеддінг """
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()

def print_results(results, title_text):
    """ Допоміжна функція для структурованого виведення результатів з Pinecone """
    print("\n" + "="*60)
    print(f"{title_text}")
    print("="*60)
    
    matches = results.get("matches", [])
    if not matches:
        print("Нічого не знайдено за вказаними критеріями фільтрації.")
        return
        
    for i, match in enumerate(matches, 1):
        meta = match.get("metadata", {})
        print(f"{i}. [Score: {match['score']:.4f}] {meta.get('title')}")
        print(f"Категорія: {meta.get('category')} | Рік: {meta.get('year')} | arXiv ID: {meta.get('arxiv_id')}")
        print(f"Абстракт : {meta.get('abstract')[:150]}...")
        print("-" * 40)

def main():
    # Чистий семантичний пошук у Pinecone (Без фільтрів)
    query_text = "teaching machines to recognize objects in pictures"
    print(f"Кодуємо запит: '{query_text}'...")
    query_vector = encode_query(query_text)

    pure_results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )
    print_results(pure_results, f"ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК ЗАПИТУ: '{query_text}'")

    # ----------------------------------------------------------------------
    # Пошук з фільтрацією (Metadata Filtering)
    # ----------------------------------------------------------------------
    
    # Приклад A: статті по reinforcement learning за останні 5 років і категорія cs.LG
    # Датасет містить статті по 2016 рік включно. 
    # Останні 5 років відносно фіналу датасету (2016) це період з 2011 по 2016 роки (year >= 2011)
    print("\nВиконуємо Приклад A: фільтрація за категорією 'cs.LG' та роком >= 2011...")
    filter_a = {
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": 2011}
    }
    results_a = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_a
    )
    print_results(results_a, "ПРИКЛАД A (Фільтр: category == cs.LG & year >= 2011)")

    # Приклад B: більш старі статті (до 2015 року включно), будь-яка категорія
    # Цей фільтр наочно продемонструє відсікання всього зрізу статей за 2016 рік
    print("\nВиконуємо Приклад B: фільтрація за роком <= 2015...")
    filter_b = {
        "year": {"$lte": 2015}
    }
    results_b = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_b
    )
    print_results(results_b, "ПРИКЛАД B (Фільтр: year <= 2015)")

    # ----------------------------------------------------------------------
    # Порівняння різних метрик схожості на локальних ембеддінгах (NumPy)
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ЛОКАЛЬНЕ ПОРІВНЯННЯ МЕТРИК СХОЖОСТІ (NumPy)")
    print("="*60)

    # Завантажуємо локальну матрицю векторів
    local_embeddings = np.load("embeddings/embeddings.npy").copy()
    local_embeddings = np.nan_to_num(local_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Перетворюємо вектор запиту у формат NumPy масиву
    q_vec = np.array(query_vector)
    q_vec = np.nan_to_num(q_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Метрика: Dot Product (Скалярний добуток)
    dot_products = local_embeddings @ q_vec
    top_5_dot = np.argsort(dot_products)[::-1][:TOP_K]

    # 2. Метрика: Cosine Similarity (Косинусна схожість)
    norm_matrix = np.linalg.norm(local_embeddings, axis=1)
    norm_query = np.linalg.norm(q_vec)
    cosine_similarities = dot_products / (norm_matrix * norm_query)
    top_5_cosine = np.argsort(cosine_similarities)[::-1][:TOP_K]

    # 3. Метрика: L2 Distance (Евклідова відстань)
    # Менша відстань = більша схожість, тому сортуємо за зростанням (без [::-1])
    l2_distances = np.linalg.norm(local_embeddings - q_vec, axis=1)
    top_5_l2 = np.argsort(l2_distances)[:TOP_K]

    # Виводимо порівняльну таблицю індексів
    print("\nТоп-5 ID статей для кожної метрики схожості:")
    metrics_comparison = pd.DataFrame({
        "Dot Product Index": [f"paper_{idx}" for idx in top_5_dot],
        "Dot Product Score": dot_products[top_5_dot],
        "Cosine Index": [f"paper_{idx}" for idx in top_5_cosine],
        "Cosine Score": cosine_similarities[top_5_cosine],
        "L2 Distance Index": [f"paper_{idx}" for idx in top_5_l2],
        "L2 Distance Score": l2_distances[top_5_l2]
    })
    print(metrics_comparison.to_string(index=False))

    # Валідація контенту: виводимо текстові метадані для найкращого збігу
    # .iloc[best_idx] строго бере рядок за його порядковим номером (0..9999), що гарантує правильну відповідність між індексами вектора та рядками DataFrame
    best_idx = top_5_dot[0]
    print(f"\nНайкращий збіг за версією Dot Product (Порядковий номер у базі: paper_{best_idx}):")
    print(f"Заголовок: {df.iloc[best_idx]['title']}")
    print(f"Категорія: {df.iloc[best_idx]['category']}")
    print(f"Рік      : {df.iloc[best_idx]['year']}")

if __name__ == "__main__":
    main()