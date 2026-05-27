"""
ОБҐРУНТУВАННЯ ЗМІНИ ЛОГІКИ ВИБІРКИ ДАНИХ:
-----------------------------------------------------------------------------
Оригінальний базовий скрипт зчитував перші 10,000 статей підряд з початку файлу.
Оскільки історичний датасет arXiv відсортований за хронологією, у перші 10,000 записів
потрапили суто статті за 2007 рік, де 99% контенту складали фундаментальна фізика,
астрономія та математика (категорії astro-ph, hep-th тощо).

Така вибірка унеможливлювала адекватне тестування Частини 3 (скрипт 04_search.py):
1. Фільтрація за категорією комп'ютерних наук (cs.LG) видавала порожні або нерелевантні
   результати через відсутність таких статей у зрізі 2007 року.
2. Фільтрація за часовими проміжками (наприклад, "за останні 5 років" чи "до 2015 року")
   не мала сенсу, бо всі документи належали до одного року.

РІШЕННЯ: Скрипт було модифіковано так, щоб він пропускав загальну фізику і збирав
перші 10,000 статей, які належать до комп'ютерних наук (префікс категорії 'cs.').
Це дозволило заглибитися в структуру JSON-файлу й назбирати статті за різні роки 
(від 2007 до 2020+), забезпечивши різноманітність даних для валідації метафільтрів.
-----------------------------------------------------------------------------
"""

import json
import os
import pandas as pd
from tqdm import tqdm

INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arxiv_subset.parquet")
MAX_RECORDS = 10000

def format_authors(paper):
    authors_list = paper.get("authors_parsed", [])
    formatted = []
    for auth in authors_list:
        if len(auth) >= 2:
            formatted.append(f"{auth[0]} {auth[1]}")
        elif len(auth) == 1:
            formatted.append(auth[0])
    return ", ".join(formatted) if formatted else paper.get("authors", "")

def extract_year(paper):
    """ Витягує рік створення статті з її ID або дат """
    paper_id = str(paper.get("id", ""))
    if "." in paper_id:
        try:
            prefix = paper_id.split(".")[0]
            year_short = int(prefix[:2])
            return 1900 + year_short if year_short > 80 else 2000 + year_short
        except ValueError:
            pass
    
    # Шукаємо в полі update_date (наприклад, "2008-11-26")
    update_date = paper.get("update_date", "")
    if update_date and len(update_date) >= 4:
        try:
            return int(update_date[:4])
        except ValueError:
            pass
    return 2007

def main():
    print(f"Починаємо сканування файлу {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Файл {INPUT_FILE} не знайдено в папці проєкту!")

    records = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # Читаємо файл рядок за рядком за допомогою tqdm
        for line in tqdm(f, desc="Сканування бази arXiv"):
            if len(records) >= MAX_RECORDS:
                break
                
            line = line.strip()
            if not line:
                continue
                
            paper = json.loads(line)
            abstract = paper.get("abstract", "").strip()
            title = paper.get("title", "").strip()

            if not abstract or not title:
                continue

            # Отримуємо основну категорію
            categories_raw = paper.get("categories", "unknown")
            primary_category = categories_raw.split()[0]

            # Шукаємо ТІЛЬКИ комп'ютерні науки (cs.*)
            if not primary_category.startswith("cs."):
                continue

            records.append({
                "id": str(paper["id"]),
                "title": title.replace("\n", " ").strip(),
                "abstract": abstract.replace("\n", " ").strip(),
                "authors": format_authors(paper),
                "year": int(extract_year(paper)),
                "category": primary_category,
            })

    # Створюємо датафрейм
    df = pd.DataFrame(records)

    print(f"\nЗавантажено релевантних cs. статей: {len(df)}")
    print("\nРозподіл за категоріями (топ-10):")
    print(df["category"].value_counts().head(10))
    print("\nРозподіл за роками:")
    print(df["year"].value_counts().sort_index())
    print("\nПриклад підготовленого запису (перший елемент):")
    if not df.empty:
        print(df.iloc[0].to_dict())
    else:
        print("Датасет порожній!")

    # Збереження
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Датасет успішно збережено в: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()