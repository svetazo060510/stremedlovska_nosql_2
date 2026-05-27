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
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# ----------------------------------------------------------------------
# Реалізація стратегій чанкінгу
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
        
        # Рухаємо покажчик вперед з урахуванням перекриття
        start += (chunk_size - overlap)
        
        # Запобігаємо нескінченному циклу або надлишковим мікро-чанкам наприкінці
        if start >= len(words) or len(words) - start <= overlap:
            break
            
    return chunks

def semantic_chunking_by_sentences(text: str, max_words: int = 60) -> list:
    """ 
    Групування цілих речень, щоб сумарний розмір чанка не перевищував max_words.
    Запобігає розриву логічних думок посередині фрази.
    """
    # Простий регулярний вираз для розбиття на речення (. ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        if not sentence_len:
            continue
            
        # Якщо одне речення саме по собі величезне, додаємо його як окремий чанк
        if sentence_len > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue
            
        # Якщо додавання речення перевищить ліміт — закриваємо поточний чанк
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
    """ Генерація ембеддінгів для чанків та завантаження батчами """
    upsert_data = []
    
    for _, row in articles_df.iterrows():
        # Викликаємо функцію нарізки
        chunks = chunk_strategy_func(row["abstract"], **kwargs)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            # Створюємо унікальний ID для кожного шматочка тексту
            unique_id = f"chunk_{strategy_name}_{row['id']}_{chunk_idx}".replace(".", "_")
            
            # Генеруємо нормалізований ембеддінг чанка
            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()
            
            # Формуємо об'єкт згідно з вимогами ДЗ
            item = (
                unique_id,
                embedding,
                {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "text": chunk_text,
                    "chunk_num": int(chunk_idx),
                    "year": int(row["year"]),
                    "category": str(row["category"])
                }
            )
            upsert_data.append(item)
            
    print(f"Завантажуємо {len(upsert_data)} чанків у Pinecone для стратегії '{strategy_name}'...")
    for i in tqdm(range(0, len(upsert_data), BATCH_SIZE), desc=f"Upsert {strategy_name}"):
        batch = upsert_data[i:i + BATCH_SIZE]
        index_client.upsert(vectors=batch)

# ----------------------------------------------------------------------
# 
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
    # Крок 1. Вибираємо 30 статей із найдовшими анотаціями
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
    
    # Стратегія 1: Fixed-size (50 слів, 10 слів перекриття)
    process_and_upload_chunks(
        top_30_longest, fixed_size_chunking, index_fixed, 
        strategy_name="fixed", chunk_size=50, overlap=10
    )
    
    # Стратегія 2: Semantic (групування речень до 60 слів)
    process_and_upload_chunks(
        top_30_longest, semantic_chunking_by_sentences, index_semantic, 
        strategy_name="semantic", max_words=60
    )

    print("\nСинхронізація індексів у хмарі...")
    
    # Крок 6. Тестові семантичні пошукові запити по чанках
    test_queries = [
        "artificial neural networks and machine learning models",
        "wireless sensor network protocols and data transmission"
    ]
    
    for query in test_queries:
        print(f"\n\nОбробимо тестовий запит: '{query}'")
        search_chunks(query, index_fixed, f"РЕЗУЛЬТАТИ ПОШУКУ ПО FIXED-SIZE ЧАНКАХ")
        search_chunks(query, index_semantic, f"РЕЗУЛЬТАТИ ПОШУКУ ПО SEMANTIC ЧАНКАХ")

if __name__ == "__main__":
    main()