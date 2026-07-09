from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.csv"
TEST_PATH = BASE_DIR / "test_queries.csv"
RESULTS_PATH = BASE_DIR / "test_results.csv"
SUMMARY_PATH = BASE_DIR / "test_summary.txt"
MIN_CONFIDENCE = 0.18


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().replace("ё", "е").strip().split())


def load_knowledge_base() -> pd.DataFrame:
    kb = pd.read_csv(KB_PATH).fillna("")
    required_columns = {"id", "topic", "question_examples", "answer", "link", "keywords"}
    missing = required_columns - set(kb.columns)
    if missing:
        raise ValueError(f"В базе знаний не хватает колонок: {', '.join(sorted(missing))}")
    kb["search_text"] = (
        kb["topic"].astype(str) + " "
        + kb["question_examples"].astype(str) + " "
        + kb["keywords"].astype(str) + " "
        + kb["answer"].astype(str)
    ).map(normalize_text)
    return kb


def build_search_index(search_texts: Tuple[str, ...]) -> Tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(search_texts)
    return vectorizer, matrix


def predict(question: str, kb: pd.DataFrame, vectorizer: TfidfVectorizer, matrix) -> dict:
    query_vector = vectorizer.transform([normalize_text(question)])
    scores = cosine_similarity(query_vector, matrix).flatten()
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    row = kb.iloc[best_idx]
    found = best_score >= MIN_CONFIDENCE
    return {
        "predicted_topic": row["topic"] if found else "Не определено",
        "confidence": round(best_score, 3),
        "found": found,
        "matched_id": row["id"],
        "matched_examples": row["question_examples"],
    }


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError("Файл test_queries.csv не найден")

    kb = load_knowledge_base()
    vectorizer, matrix = build_search_index(tuple(kb["search_text"].tolist()))
    tests = pd.read_csv(TEST_PATH).fillna("")

    required_columns = {"question", "expected_topic"}
    missing = required_columns - set(tests.columns)
    if missing:
        raise ValueError(f"В test_queries.csv не хватает колонок: {', '.join(sorted(missing))}")

    rows = []
    for _, test in tests.iterrows():
        result = predict(test["question"], kb, vectorizer, matrix)
        rows.append({
            "question": test["question"],
            "expected_topic": test["expected_topic"],
            "predicted_topic": result["predicted_topic"],
            "is_correct": test["expected_topic"] == result["predicted_topic"],
            "confidence": result["confidence"],
            "found": result["found"],
            "matched_id": result["matched_id"],
            "matched_examples": result["matched_examples"],
        })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    total = len(results)
    correct = int(results["is_correct"].sum())
    accuracy = correct / total * 100 if total else 0
    avg_confidence = results["confidence"].mean() if total else 0
    found_share = results["found"].mean() * 100 if total else 0

    per_topic = (
        results.groupby("expected_topic")["is_correct"]
        .agg(["count", "sum"])
        .reset_index()
    )
    per_topic["accuracy_percent"] = (per_topic["sum"] / per_topic["count"] * 100).round(1)

    errors = results[~results["is_correct"]].copy()

    lines = [
        "Результаты автотестирования Telegram-бота 'Помощник водителю'",
        f"Всего тестовых запросов: {total}",
        f"Правильно определена тема: {correct}",
        f"Accuracy: {accuracy:.1f}%",
        f"Средняя уверенность: {avg_confidence:.3f}",
        f"Доля найденных ответов: {found_share:.1f}%",
        "",
        "Точность по темам:",
    ]
    for _, row in per_topic.iterrows():
        lines.append(
            f"- {row['expected_topic']}: {int(row['sum'])}/{int(row['count'])} "
            f"({row['accuracy_percent']:.1f}%)"
        )

    if not errors.empty:
        lines += ["", "Ошибки:"]
        for _, row in errors.iterrows():
            lines.append(
                f"- '{row['question']}' | ожидалось: {row['expected_topic']} | "
                f"получено: {row['predicted_topic']} | confidence={row['confidence']}"
            )
    else:
        lines += ["", "Ошибок на тестовом наборе нет."]

    summary = "\n".join(lines)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nПодробные результаты сохранены в {RESULTS_PATH.name}")
    print(f"Краткий отчет сохранен в {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
