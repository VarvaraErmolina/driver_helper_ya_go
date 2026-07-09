from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.csv"
LOG_PATH = BASE_DIR / "telegram_user_logs.csv"
MIN_CONFIDENCE = 0.18

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().replace("ё", "е").strip().split())


def load_knowledge_base() -> pd.DataFrame:
    if not KB_PATH.exists():
        raise FileNotFoundError("Файл knowledge_base.csv не найден")

    kb = pd.read_csv(KB_PATH)
    required_columns = {"id", "topic", "question_examples", "answer", "link", "keywords"}
    missing = required_columns - set(kb.columns)
    if missing:
        raise ValueError(f"В базе знаний не хватает колонок: {', '.join(sorted(missing))}")

    kb = kb.fillna("")
    kb["search_text"] = (
        kb["topic"].astype(str) + " " +
        kb["question_examples"].astype(str) + " " +
        kb["keywords"].astype(str) + " " +
        kb["answer"].astype(str)
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


KB = load_knowledge_base()
VECTORIZER, MATRIX = build_search_index(tuple(KB["search_text"].tolist()))


def find_answer(question: str) -> Dict[str, object]:
    query_vector = VECTORIZER.transform([normalize_text(question)])
    scores = cosine_similarity(query_vector, MATRIX).flatten()

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    row = KB.iloc[best_idx]
    found = best_score >= MIN_CONFIDENCE

    if found:
        answer = row["answer"]
        topic = row["topic"]
        link = row["link"]
    else:
        answer = (
            "Я не нашел точный ответ в базе знаний. Лучше передать вопрос в поддержку. "
            "Опишите ситуацию, дату заказа или выплаты, город и приложите скриншот, если он есть."
        )
        topic = "Не определено"
        link = ""

    return {
        "found": found,
        "topic": topic,
        "answer": answer,
        "link": link,
        "confidence": round(best_score, 3),
        "matched_id": row["id"],
        "matched_question_examples": row["question_examples"],
    }


def ensure_log_header() -> None:
    if LOG_PATH.exists():
        return
    with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "user_id",
            "username",
            "question",
            "topic",
            "confidence",
            "found",
            "matched_id",
        ])


def save_log(update: Update, question: str, result: Dict[str, object]) -> None:
    ensure_log_header()
    user = update.effective_user
    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user.id if user else "",
            user.username if user and user.username else "",
            question,
            result["topic"],
            result["confidence"],
            result["found"],
            result["matched_id"],
        ])


def format_answer(result: Dict[str, object]) -> str:
    status = "✅ Ответ найден" if result["found"] else "⚠️ Точного ответа нет в базе знаний"
    confidence_pct = int(float(result["confidence"]) * 100)

    text = (
        f"{status}\n\n"
        f"*Тема:* {result['topic']}\n"
        f"*Уверенность:* {confidence_pct}%\n\n"
        f"*Ответ:*\n{result['answer']}"
    )

    if result["link"]:
        text += f"\n\n[Открыть инструкцию]({result['link']})"

    text += "\n\nЕсли ситуация срочная или связана с безопасностью, обратитесь в поддержку платформы."
    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет! Я *Помощник водителю* 🚕\n\n"
        "Я отвечаю на частые вопросы о выплатах, рейтинге, блокировках, документах, штрафах, заказах и безопасности.\n\n"
        "Задайте мне вопрос обычными словами.\n"
        "Команды: /topics — темы, /stats — статистика прототипа, /help — помощь."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Как пользоваться ботом:\n\n"
        "1. Напишите вопрос свободным текстом.\n"
        "2. Бот определит тему и найдет похожий ответ в базе знаний.\n"
        "3. Если точного ответа нет, бот предложит обратиться в поддержку.\n\n"
        "Для демонстрации попробуйте запросы:\n"
        "• Когда придут деньги?\n"
        "• Почему списали деньги?\n"
        "• Не могу выйти на линию\n"
        "• Клиент угрожает, что делать?"
    )
    await update.message.reply_text(text)


async def topics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    topic_list = sorted(KB["topic"].dropna().unique().tolist())
    text = "Я умею отвечать по темам:\n\n" + "\n".join(f"• {topic}" for topic in topic_list)
    await update.message.reply_text(text)


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not LOG_PATH.exists():
        await update.message.reply_text("Пока нет сохраненных обращений.")
        return

    logs = pd.read_csv(LOG_PATH)
    if logs.empty:
        await update.message.reply_text("Пока нет сохраненных обращений.")
        return

    top_topics = logs["topic"].value_counts().head(5)
    found_share = logs["found"].astype(bool).mean() * 100
    lines = [
        "📊 Статистика прототипа",
        f"Всего вопросов: {len(logs)}",
        f"Доля найденных ответов: {found_share:.0f}%",
        "",
        "Частые темы:",
    ]
    lines += [f"• {topic}: {count}" for topic, count in top_topics.items()]
    await update.message.reply_text("\n".join(lines))


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text.strip()
    if not question:
        await update.message.reply_text("Напишите вопрос текстом.")
        return

    result = find_answer(question)
    save_log(update, question, result)
    await update.message.reply_text(
        format_answer(result),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Не найден TELEGRAM_BOT_TOKEN. Создайте файл .env или задайте переменную окружения."
        )

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("topics", topics))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Telegram bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
