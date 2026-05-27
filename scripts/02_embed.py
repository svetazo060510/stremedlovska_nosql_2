import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Шляхи до файлів
INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")

# Налаштування моделі
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

def main():
    # 1. Завантаження датасету
    print(f"Зчитуємо датасет з {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Файл {INPUT_FILE} не знайдено! Спочатку запустіть скрипт 01_prepare_data.py")
    
    df = pd.read_parquet(INPUT_FILE)
    
    # 2. Підготовка текстів для кодування (title + [SEP] + abstract)
    print("Форматуємо тексти для моделі specter2...")
    # Обов'язково використовуємо токен [SEP], як вимагає картка моделі
    texts_to_encode = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    
    # 3. Ініціалізація моделі
    print(f"Завантажуємо модель {MODEL_NAME} з HuggingFace (це може зайняти певний час при першому запуску)...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Генерація ембеддінгів
    print(f"Починаємо генерацію ембеддінгів (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        sentences=texts_to_encode,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,       # Увімкнення відображення прогресу
        normalize_embeddings=True     # Нормалізація ембеддінгів (L2 нормалізація)
    )
    
    # 5. Валідація та вивід інформації в консоль
    num_processed = len(embeddings)
    embedding_dim = embeddings.shape[1]
    
    # Рахуємо норму першого вектора за формулою Евклідової норми
    first_vector_norm = np.linalg.norm(embeddings[0])
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТИ КОДУВАННЯ:")
    print(f"Загальна кількість оброблених текстів : {num_processed}")
    print(f"Розмірність отриманих ембеддінгів     : {embedding_dim} (Очікувалось: 768)")
    print(f"Норма першого ембеддінгу              : {first_vector_norm:.4f} (Очікувалось: ~1.0)")
    print("="*50 + "\n")
    
    # 6 & 7. Збереження результатів та створення директорії за потреби
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Зберігаємо матрицю ембеддінгів у {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, embeddings)
    print("Скрипт успішно завершив роботу!")

if __name__ == "__main__":
    main()