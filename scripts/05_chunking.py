import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 100

# Ініціалізація інструментів
if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("PINECONE_API_KEY не знайдено у файлі .env!")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# ----------------------------------------------------------------------
# Реалізація чанкінгу 
# ----------------------------------------------------------------------

def fixed_size_chunking(text: str, chunk_size: int = 50, overlap: int = 10) -> list:
    """ Нарізка на фіксовану кількість слів із перекриттям """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        start += (chunk_size - overlap)
        if start >= len(words) or len(words) - start <= overlap:
            break
            
    return chunks

def semantic_chunking_by_sentences(text: str, max_words: int = 60) -> list:
    """ Групування цілих речень з обмеженням розміру чанка """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        if not sentence_len:
            continue
            
        if sentence_len > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue
            
        if current_word_count + sentence_len > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = sentence_words
            current_word_count = sentence_len
        else:
            current_chunk.extend(sentence_words)
            current_word_count += sentence_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# ----------------------------------------------------------------------
# Ініціалізація індексів у Pinecone
# ----------------------------------------------------------------------

def init_pinecone_index(index_name: str):
    """ Створення серверного індексу, якщо він не існує """
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Створюємо індекс '{index_name}' у Pinecone...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Індекс '{index_name}' успішно створено.")
    else:
        print(f"Індекс '{index_name}' вже існує.")
    return pc.Index(index_name)

# ----------------------------------------------------------------------
# Обробка та завантаження чанків
# ----------------------------------------------------------------------

def process_and_upload_chunks(articles_df: pd.DataFrame, chunk_strategy_func, index_client, strategy_name: str, **kwargs):
    """ Генерація ембеддінгів методом пакетної обробки (Батчингу) """
    raw_chunks_info = []
    
    # Крок 1: Тільки нарізаємо тексти (швидка операція в пам'яті)
    records = articles_df.to_dict(orient="records")
    for row in records:
        chunks = chunk_strategy_func(row["abstract"], **kwargs)
        for chunk_idx, chunk_text in enumerate(chunks):
            # Заміна крапок на підкреслення є обов'язковою, оскільки синтаксичні правила 
            # формування системних ID у Pinecone API мають жорсткі обмеження на спецсимволи.
            # Щоб уникнути "silently corrupt IDs", оригінальний arXiv ID з крапкою 
            # зберігається всередині словника metadata під ключем "arxiv_id".
            unique_id = f"chunk_{strategy_name}_{row['id']}_{chunk_idx}".replace(".", "_")
            raw_chunks_info.append({
                "id": unique_id,
                "text": chunk_text,
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "text": chunk_text,
                    "chunk_num": int(chunk_idx),
                    "year": int(row["year"]),
                    "category": str(row["category"])
                }
            })
            
    total_chunks = len(raw_chunks_info)
    if total_chunks == 0:
        print(f"Немає чанків для стратегії {strategy_name}.")
        return

    # Крок 2: Пакетна генерація ембеддінгів для всіх чанків одним викликом
    print(f"Генеруємо ембеддінги для {total_chunks} чанків стратегії '{strategy_name}' (Пакетний режим)...")
    texts_to_encode = [c["text"] for c in raw_chunks_info]
    
    # Використовуємо пакетний encode
    embeddings = model.encode(
        sentences=texts_to_encode, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    # Крок 3: Формуємо фінальний масив кортежів для Pinecone
    upsert_data = []
    for i, chunk in enumerate(raw_chunks_info):
        item = (
            chunk["id"],
            embeddings[i].tolist(),
            chunk["metadata"]
        )
        upsert_data.append(item)
            
    # Крок 4: Завантаження батчами у Pinecone
    print(f"Завантажуємо {len(upsert_data)} чанків у Pinecone індекс...")
    for i in tqdm(range(0, len(upsert_data), BATCH_SIZE), desc=f"Upsert {strategy_name}"):
        batch = upsert_data[i:i + BATCH_SIZE]
        index_client.upsert(vectors=batch)

# ----------------------------------------------------------------------
# Функція пошуку по чанках
# ----------------------------------------------------------------------

def search_chunks(query: str, index_client, title_text: str):
    """ Пошук по чанках та виведення результатів """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = index_client.query(vector=query_vector, top_k=5, include_metadata=True)
    
    print("\n" + "="*70)
    print(f"{title_text}")
    print("="*70)
    for i, match in enumerate(results.get("matches", []), 1):
        meta = match.get("metadata", {})
        print(f"{i}. [Score: {match['score']:.4f}] Стаття: {meta.get('title')}")
        print(f"ID: {meta.get('arxiv_id')} | Чанк №{meta.get('chunk_num')} | Категорія: {meta.get('category')}")
        print(f"Текст чанка: {meta.get('text')[:180]}...")
        print("-" * 50)

# ----------------------------------------------------------------------
# Головний конвеєр
# ----------------------------------------------------------------------

def main():
    # Вибираємо 30 статей із найдовшими анотаціями
    print("Аналізуємо довжину анотацій та відбираємо топ-30 найдовших статей...")
    df["abstract_len"] = df["abstract"].apply(lambda x: len(str(x).split()))
    top_30_longest = df.sort_values(by="abstract_len", ascending=False).head(30)
    
    print(f"Мінімальна довжина відібраних анотацій: {top_30_longest['abstract_len'].min()} слів.")
    print(f"Максимальна довжина відібраних анотацій: {top_30_longest['abstract_len'].max()} слів.")

    # Ініціалізуємо індекси у хмарі Pinecone
    index_fixed = init_pinecone_index("arxiv-chunks-fixed")
    index_semantic = init_pinecone_index("arxiv-chunks-semantic")

    # Обробка та завантаження для обох стратегій
    print("\n--- Запуск чанкінгу та генерації ембеддінгів ---")
    
    # Fixed-size (50 слів, 10 слів перекриття)
    process_and_upload_chunks(
        top_30_longest, fixed_size_chunking, index_fixed, 
        strategy_name="fixed", chunk_size=50, overlap=10
    )
    
    # Semantic (групування речень до 60 слів)
    process_and_upload_chunks(
        top_30_longest, semantic_chunking_by_sentences, index_semantic, 
        strategy_name="semantic", max_words=60
    )

    print("\nСинхронізація індексів у хмарі...")
    
    # Тестові семантичні пошукові запити по чанках
    test_queries = [
        "artificial neural networks and machine learning models",
        "wireless sensor network protocols and data transmission"
    ]
    
    for query in test_queries:
        print(f"\n\nОбробляємо тестовий запит: '{query}'")
        search_chunks(query, index_fixed, f"РЕЗУЛЬТАТИ ПОШУКУ ПО FIXED-SIZE ЧАНКАХ")
        search_chunks(query, index_semantic, f"РЕЗУЛЬТАТИ ПОШУКУ ПО SEMANTIC ЧАНКАХ")

if __name__ == "__main__":
    main()