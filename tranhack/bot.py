import logging
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

from tranhack.generate_answers import LLMGenerator

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

NAME, FACULTY, DIRECTION = range(3)


class Bot():
    def __init__(self):
        self.gen = LLMGenerator()

    def make_translation(self, name, faculty, direction):
        return self.gen.generate([{'name': name, 'faculty': faculty, 'direction': direction}])[0]


    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = [
            [InlineKeyboardButton("Start Translation", callback_data="start_translation")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "Приветствуем! Я помогу вам перевести названия научных работ на английский язык",
            reply_markup=reply_markup
        )

        return NAME


    async def start_translation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("Пожалуйста, введите название работы:")

        return NAME


    async def collect_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data['name'] = update.message.text
        await update.message.reply_text("Принято! Пожалуйста, введите факультет:")

        return FACULTY


    async def collect_faculty(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data['faculty'] = update.message.text
        await update.message.reply_text("Спасибо! Введите направление:")

        return DIRECTION


    async def collect_direction(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data['direction'] = update.message.text

        name = context.user_data.get('name')
        faculty = context.user_data.get('faculty')
        direction = context.user_data.get('direction')

        translated_text = (
            f"Перевод Статьи:\n"
            f"Имя: <b>{name}</b>. Факультет: {faculty}. Направление: {direction}\n"
            f"Перевод: <b>{self.make_translation(name, faculty, direction)}</b>"
        )

        keyboard = [
            [InlineKeyboardButton("New Translation", callback_data="start_translation")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(translated_text, parse_mode='HTML')
        await update.message.reply_text(
            "Хотели бы сделать еще перевод?", reply_markup=reply_markup
        )

        return ConversationHandler.END


    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await update.message.reply_text("Операция отменена. Введите /start, чтобы начать сначала.")
        return ConversationHandler.END


    def run(self):
        TOKEN = os.getenv('TG_TOKEN')

        application = Application.builder().token(TOKEN).build()

        conv_handler = ConversationHandler(
            entry_points=[CallbackQueryHandler(self.start_translation, pattern="^start_translation$")],
            states={
                NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.collect_name)],
                FACULTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.collect_faculty)],
                DIRECTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.collect_direction)],
            },
            fallbacks=[CommandHandler("cancel", self.cancel)],
        )

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(conv_handler)

        application.run_polling()


if __name__ == "__main__":
    Bot().run()
