from nlp import generate_answer
from loader import BOT_TOKEN

import logging
from telegram import Update, Bot, User
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

logging.basicConfig(level=logging.INFO)


def start_command(update: Update, context: CallbackContext) -> None:
    logging.info("Triggered command '/start'")
    update.message.reply_text('Hi!')


def help_command(update: Update, context: CallbackContext) -> None:
    logging.info("Triggered command '/help'")
    update.message.reply_text('Help!')


def get_reply(update: Update, context: CallbackContext) -> None:
    replica = update.message.text
    logging.info(f"Got message with text: {replica}")
    answer = generate_answer(replica)
    update.message.reply_text(answer)


def start():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(BOT_TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, get_reply))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    start()
