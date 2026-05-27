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
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

def encode_query(query: str) -> list:
    """ Крок 2. Функція кодування запиту в нормалізований ембеддінг """
    # Модель specter2 очікує той самий формат, але без abstract, якщо це запит
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()

def print_results(results, title_text):
    """ Допоміжна функція для красивого виведення результатів Pinecone """
    print("\n" + "="*60)
    print(f"{title_text}")
    print("="*60)
    for i, match in enumerate(results.get("matches", []), 1):
        meta = match.get("metadata", {})
        print(f"{i}. [Score: {match['score']:.4f}] {meta.get('title')}")
        print(f"Категорія: {meta.get('category')} | Рік: {meta.get('year')} | arXiv ID: {meta.get('arxiv_id')}")
        print(f"Абстракт : {meta.get('abstract')[:150]}...")
        print("-" * 40)

def main():
    # Текстовий запит користувача
    query_text = "teaching machines to recognize objects in pictures"
    print(f"Кодуємо запит: '{query_text}'...")
    query_vector = encode_query(query_text)

    # ----------------------------------------------------------------------
    # Чистий семантичний пошук у Pinecone
    # ----------------------------------------------------------------------
    pure_results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )
    print_results(pure_results, f"ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК ЗАПИТУ: '{query_text}'")

    # ----------------------------------------------------------------------
    # Пошук з фільтрацією (Metadata Filtering)
    # ----------------------------------------------------------------------
    # Приклад A: Reinforcement learning за останні 5 років (відносно 2007 року в нашому датасеті) і категорія cs.LG
    # Наш датасет містить статті ТІЛЬКИ за 2007 рік, тому "останні 5 років" зробимо як >= 2005
    print("\nВиконуємо Приклад A: фільтрація за категорією 'cs.LG' та роком >= 2005...")
    filter_a = {
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": 2005}
    }
    results_a = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_a
    )
    print_results(results_a, "ПРИКЛАД A (Фільтр: category == cs.LG & year >= 2005)")

    # Приклад B: Більш старі статті (до 2015 року), будь-яка категорія
    # Оскільки у нас всі статті 2007 року, фільтр year <= 2015 поверне результати
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
    # Порівняння різних метрик схожості на локальних ембеддінгах
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ЛОКАЛЬНЕ ПОРІВНЯННЯ МЕТРИК СХОЖОСТІ (NumPy)")
    print("="*60)

    # 1. Завантажуємо матрицю і робимо її копію, яку можна редагувати (.copy())
    local_embeddings = np.load("embeddings/embeddings.npy").copy()

    # 2. Очищаємо копію (тепер copy=True або просто без copy=False)
    local_embeddings = np.nan_to_num(local_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Готуємо і очищаємо вектор запиту
    q_vec = np.array(query_vector)
    q_vec = np.nan_to_num(q_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Dot Product (Скалярний добуток)
    # Для матриці 10000x768 та вектора 768 скалярний добуток рахується через матричне множення @
    dot_products = local_embeddings @ q_vec
    top_5_dot = np.argsort(dot_products)[::-1][:TOP_K]

    # 2. Cosine Similarity
    # Оскільки і матриця, і вектор вже нормалізовані (L2 норма = 1), косинус повністю дорівнює Dot Product!
    # Проте пропишемо чесну математичну формулу для демонстрації викладачу:
    norm_matrix = np.linalg.norm(local_embeddings, axis=1)
    norm_query = np.linalg.norm(q_vec)
    cosine_similarities = dot_products / (norm_matrix * norm_query)
    top_5_cosine = np.argsort(cosine_similarities)[::-1][:TOP_K]

    # 3. L2 Distance (Евклідова відстань)
    # Чим МЕНША відстань, тим БЛИЖЧІ вектори. Тому сортуємо по зростанню (без [::-1])
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

    # Виведемо назву найкращої статті для перевірки релевантності змісту
    best_idx = top_5_dot[0]
    print(f"\nНайкращий збіг за версією Dot Product (ID: paper_{best_idx}):")
    print(f"Заголовок: {df.iloc[best_idx]['title']}")
    print(f"Категорія: {df.iloc[best_idx]['category']}")

if __name__ == "__main__":
    main()