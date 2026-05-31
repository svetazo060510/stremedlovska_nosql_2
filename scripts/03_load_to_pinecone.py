import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Завантажуємо змінні оточення з .env
load_dotenv()

# Константи та шляхи
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Оптимальний розмір батчу для Pinecone

def main():
    # Перевірка наявності API ключа
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY не знайдено у файлі .env! Перевірте налаштування.")

    # Ініціалізація клієнта Pinecone
    pc = Pinecone(api_key=api_key)

    # 1. Створюємо індекс arxiv-papers, якщо він ще не існує
    print(f"Перевіряємо наявність індексу '{INDEX_NAME}'...")
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Індекс '{INDEX_NAME}' не знайдено. Створюємо новий...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="dotproduct", # Використовуємо Dot Product, бо вектори нормалізовані
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Стандартний безкоштовний регіон
            )
        )
        print(f"Індекс '{INDEX_NAME}' успішно створено!")
    else:
        print(f"Індекс '{INDEX_NAME}' вже існує.")

    # Підключаємося до індексу
    index = pc.Index(INDEX_NAME)

    # 2. Завантаження локальних даних
    print(f"\nЗчитуємо дані з {INPUT_PARQUET} та {INPUT_EMBEDDINGS}...")
    if not os.path.exists(INPUT_PARQUET) or not os.path.exists(INPUT_EMBEDDINGS):
        raise FileNotFoundError("Бракує файлів даних або ембеддінгів! Перевірте кроки 01 та 02.")

    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    # 3 & 4. Підготовка даних та завантаження в Pinecone батчами на льоту
    total_records = len(df)
    print(f"\nФормуємо об'єкти та завантажуємо {total_records} векторів у Pinecone батчами по {BATCH_SIZE}...")

    # Перетворюємо DataFrame на список словників для значного прискорення ітерації
    records = df.to_dict(orient="records")
    
    current_batch = []
    
    # Використовуємо tqdm з фіксованим total для відображення прогресу батчів
    with tqdm(total=total_records, desc="Завантаження в хмару") as pbar:
        # enumerate(..., start=0) гарантує правильну послідовну індексацію векторів
        for i, row in enumerate(records):
            clean_abstract = str(row["abstract"])[:500]
            clean_authors = str(row["authors"])[:200]
            
            # Структуруємо елемент для Pinecone
            item = (
                f"paper_{i}",                      # Послідовний ID: paper_0, paper_1 ... paper_9999
                embeddings[i].tolist(),            # Точна відповідність рядку в матриці npy
                {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "abstract": clean_abstract,
                    "authors": clean_authors,
                    "year": int(row["year"]),
                    "category": str(row["category"])
                }
            )
            current_batch.append(item)
            
            # Якщо назбирали повний батч або це фінальний елемент — робимо upsert
            if len(current_batch) == BATCH_SIZE or i == total_records - 1:
                index.upsert(vectors=current_batch)
                pbar.update(len(current_batch))    # Оновлюємо прогрес-бар на кількість надісланих векторів
                current_batch = []                 # Очищуємо батч для економії пам'яті

    # 5. Перевірка фінального статусу індексу
    print("\nСинхронізація індексу...")
    index_stats = index.describe_index_stats()
    
    print("\n" + "="*50)
    print("СТАТИСТИКА ХМАРНОГО ІНДЕКСУ:")
    print(f"Загальна кількість векторів в індексі: {index_stats['total_vector_count']}")
    print("="*50 + "\n")
    print("Завантаження повністю завершено і оптимізовано за пам'яті!")

if __name__ == "__main__":
    main()