import os
import torch
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
    
    # --- ОПТИМІЗАЦІЯ: ДИНАМІЧНИЙ ПАДІНГ (Dynamic Padding через сортування) ---
    # Запам'ятовуємо оригінальний порядок, сортуємо тексти за довжиною (скорочує обчислення падінгів)
    print("Застосовуємо Dynamic Padding (сортування текстів для оптимізації обчислень на будь-якому CPU/GPU)...")
    sorted_indices = np.argsort([len(t) for t in texts_to_encode])
    sorted_texts = [texts_to_encode[idx] for idx in sorted_indices]
    
    # --- КРОСПЛАТФОРМНЕ АВТО-ВИЗНАЧЕННЯ ЗАЛІЗА ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Автовизначення середовища виконання. Обчислення будуть запущені на: [{device.upper()}]")

    # 3. Ініціалізація моделі
    print(f"Завантажуємо модель {MODEL_NAME} на пристрій [{device}]...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # 4. Генерація ембеддінгів (над сортованим масивом)
    print(f"Починаємо генерацію ембеддінгів (batch_size={BATCH_SIZE})...")
    sorted_embeddings = model.encode(
        sentences=sorted_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True  # Нормалізація ембеддінгів (L2 нормалізація)
    )
    
    # --- ВІДНОВЛЕННЯ ПОРЯДКУ ДАНІХ ---
    # Повертаємо вектори у вихідний порядок, щоб вони чітко відповідали рядкам у Parquet-файлі
    print("Реконструюємо початковий порядок векторів для збереження цілісності даних...")
    inv_indices = np.argsort(sorted_indices)
    embeddings = sorted_embeddings[inv_indices]
    
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
    print("Скрипт успішно завершив роботу у безпечному кросплатформному режимі!")

if __name__ == "__main__":
    main()