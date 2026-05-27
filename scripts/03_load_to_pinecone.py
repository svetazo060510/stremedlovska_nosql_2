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
                region="us-east-1"  # Стандартний безкоштовний регіон (Starter Tier)
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

    # 3. Підготовка даних для завантаження
    print("Формуємо структуровані об'єкти (id, vectors, metadata)...")
    upsert_data = []
    
    for idx, row in df.iterrows():
        # Обрізаємо абстракт та авторів згідно з вимогами ДЗ
        clean_abstract = str(row["abstract"])[:500]
        clean_authors = str(row["authors"])[:200]
        
        # Pinecone очікує кортеж: (id, vector, metadata)
        item = (
            f"paper_{idx}",                                 # Унікальний id вигляду paper_0, paper_1...
            embeddings[idx].tolist(),                       # Перетворюємо масив NumPy у звичайний список Python float
            {                                               # Словник метаданих
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": clean_abstract,
                "authors": clean_authors,
                "year": int(row["year"]),
                "category": str(row["category"])
            }
        )
        upsert_data.append(item)

    # 4. Завантаження даних в Pinecone батчами з прогрес-баром
    print(f"\nЗавантажуємо {len(upsert_data)} векторів у Pinecone батчами по {BATCH_SIZE}...")
    
    # Крокуємо по всьому масиву з кроком BATCH_SIZE
    for i in tqdm(range(0, len(upsert_data), BATCH_SIZE), desc="Завантаження в хмару"):
        batch = upsert_data[i:i + BATCH_SIZE]
        # Команда upsert виконує вставку або оновлення записів
        index.upsert(vectors=batch)

    # 5. Перевірка фінального статусу індексу
    print("\nСинхронізація індексу...")
    index_stats = index.describe_index_stats()
    
    print("\n" + "="*50)
    print("СТАТИСТИКА ХМАРНОГО ІНДЕКСУ:")
    print(f"Загальна кількість векторів в індексі: {index_stats['total_vector_count']}")
    print("="*50 + "\n")
    print("Завантаження повністю завершено!")

if __name__ == "__main__":
    main()