import json
import os
import pandas as pd
from tqdm import tqdm

"""
===================================================================================
📌 АРХІТЕКТУРНІ ЗМІНИ ТА ОБҐРУНТУВАННЯ (ПОРІВНЯННО З ОРИГІНАЛЬНИМ СКРИПТОМ З ДЗ):
===================================================================================
1. ЩО ЗМІНЕНО: 
   Додано жорстку фільтрацію за галуззю знань `primary_category.startswith("cs.")` 
   та впроваджено систему порічних квот (Stratified Sampling) через словник `year_counts`.

2. ЧОМУ ЦЕ БУЛО НЕОБХІДНО:
   - В оригінальному скрипті ДЗ код просто читав перші 10 000 рядків файлу підряд. 
     Оскільки архів arXiv відсортований хронологічно, а на початку його створення 
     там публікувалася переважно фізика, оригінальний скрипт збирав виключно статті 
     з рідкої фізики та математики строго за один 2007 рік.
   - Через це в наступних скриптах повністю ламалася логіка метафільтрації Pinecone. 
     Запити на кшталт "статті з машинного навчання (cs.LG) за роками <= 2015" 
     повертали порожні результати (0 знайдених статей).

3. РЕЗУЛЬТАТ МОДИФІКАЦІЇ:
   Код повністю зберігає оригінальну структуру обробки тексту, очищення рядків 
   та парсингу авторів з ДЗ, але тепер гарантовано збирає професійний IT-датасет: 
   рівно по 1 000 релевантних статей з комп'ютерних наук для кожного року з 2007 по 2016.
===================================================================================
"""

INPUT_FILE  = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"

# --- МОДИФІКАЦІЯ: НАЛАШТУВАННЯ СТРАТИФІКОВАНОЇ ВИБІРКИ ЗА РОКАМИ ---
YEARS_TO_COLLECT = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
LIMIT_PER_YEAR = 1000
TOTAL_MAX_RECORDS = len(YEARS_TO_COLLECT) * LIMIT_PER_YEAR  # Сумарно: 10 000 статей

# Словник для контролю лімітів: {2007: 0, 2008: 0 ... 2016: 0}
year_counts = {year: 0 for year in YEARS_TO_COLLECT}
# -------------------------------------------------------------------

os.makedirs("data", exist_ok=True)

def extract_year(paper: dict) -> int:
    """ Витягуємо справжній рік публікації з першої версії """
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]  # "Mon, 2 Apr 2007 19:18:42 GMT"
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    return int(paper.get("update_date", "2000-01-01")[:4])

def format_authors(paper: dict) -> str:
    """ Парсинг структурованого списку авторів (макс 10) """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return paper.get("authors", "").replace("\\n", " ")

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Сканування бази arXiv"):
        # Зупиняємося, коли повністю заповнили всі порічні квоти (10 000 статей)
        if len(records) >= TOTAL_MAX_RECORDS:
            break
            
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = paper.get("abstract", "").strip()
        title    = paper.get("title", "").strip()

        if not abstract or not title:
            continue

        # Виділяємо основну категорію
        categories_raw = paper.get("categories", "unknown")
        primary_category = categories_raw.split()[0]

        # --- ПЕРЕВІРКА ГАЛУЗІ ТА ПОРІЧНИХ КВOТ ---
        # 1. Відбираємо виключно комп'ютерні науки (Computer Science)
        if not primary_category.startswith("cs."):
            continue
            
        # 2. Визначаємо справжній рік публікації
        paper_year = extract_year(paper)
        
        # 3. Перевіряємо, чи цей рік нам потрібен і чи є ще вільне місце в квоті
        if paper_year in year_counts and year_counts[paper_year] < LIMIT_PER_YEAR:
            # Фіксуємо заповнення квоти для цього року
            year_counts[paper_year] += 1
            
            # Додаємо запис до результатів
            records.append({
                "id":       paper["id"],
                "title":    title.replace("\\n", " ").strip(),
                "abstract": abstract.replace("\\n", " ").strip(),
                "authors":  format_authors(paper),
                "year":     paper_year,
                "category": primary_category,
            })
        # -------------------------------------------------------------

# --- ФІНАЛЬНИЙ БЛОК ВИВЕДЕННЯ СТАТИСТИКИ  ---
df = pd.DataFrame(records)
print(f"\nЗавантажено статей: {len(df)}")
print(f"\nРозподіл за категоріями (топ-10):")
print(df["category"].value_counts().head(10))
print(f"\nРозподіл за роками:")
print(df["year"].value_counts().sort_index().tail(10))
print(f"\nПриклад запису:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nЗбережено в {OUTPUT_FILE}")