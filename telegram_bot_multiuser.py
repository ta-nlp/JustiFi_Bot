from telegram import Update, Document
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os
import glob
import uuid
from utils.retrieve_and_rerank import create_bm25_index
from utils.upload_and_embedd import upload_files
from utils.generate_ans import query_model
from utils.classify_query import classify_query
from dotenv import load_dotenv
import logging
from telegram.constants import ChatAction
import traceback
from utils.rewrite_query import process_query


# Load environment variables from .env file
load_dotenv()

# Ensure uploads directory exists
UPLOAD_FOLDER = "./uploads"
LOGS_FOLDER = "./logs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Dictionary to store user-specific session data
user_sessions = {}

#Configure Logger
def get_user_logger(user_id, user_name):
    """Creates or retrieves a user-specific logger."""
    log_filename = os.path.join(LOGS_FOLDER, f"user_{user_id}_{user_name}.log")
    user_logger = logging.getLogger(str(user_id))

    if not user_logger.hasHandlers():  # Prevent duplicate handlers
        handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        user_logger.addHandler(handler)
        user_logger.setLevel(logging.INFO)

         # 🚀 Force a log entry to ensure the file is created
        user_logger.info("Logger initialized successfully.")
    
    return user_logger



#Function to handle input file and make embedding
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id  # Unique user ID
    user_name = update.effective_user.username
    document: Document = update.message.document

    # logger.info(f"User {update.effective_user.id} - {update.effective_user.username} uploaded file: {document.file_name}")
    # Notify user that file upload is in progress
    waiting_message = await update.message.reply_text("Uploading and processing your file... Please wait.")

    try:
            # Generate a unique filename using UUID
            unique_id = uuid.uuid4().hex
            file_extension = os.path.splitext(document.file_name)[1]  # Get file extension
            unique_filename = f"{unique_id}{file_extension}"
            user_upload_folder = os.path.join(UPLOAD_FOLDER, str(user_id))
            os.makedirs(user_upload_folder, exist_ok=True)  # Create user-specific folder
            file_path = os.path.join(user_upload_folder, unique_filename)
            
            # Download file
            file = await document.get_file()
            await file.download_to_drive(file_path)

            # Process the uploaded file
            cleaned_texts, vector_store = upload_files([file_path], 1000, 100)
            bm25 = create_bm25_index(cleaned_texts)

            # Store session data for the user
            user_sessions[user_id] = {
                "vector_store": vector_store,
                "cleaned_texts": cleaned_texts,
                "bm25": bm25,
                "upload_folder": user_upload_folder,
                "conversation_history": []
            }

            os.remove(file_path)
            user_logger = get_user_logger(user_id, user_name)
            user_logger.info(f"File '{document.file_name}' processed successfully for user {update.effective_user.id} - {update.effective_user.username}.")
            # logger.info(f"File '{document.file_name}' processed successfully for user {update.effective_user.id}.")
            await waiting_message.edit_text(f"File '{document.file_name}' uploaded and processed successfully! You can now ask queries.")

    except Exception as e:
            # logger.error(f"Error processing file {document.file_name} for user {update.effective_user.id}: {str(e)}")
            await waiting_message.edit_text("Failed to process the file. Please try again.")




# Function to run the query model and get the answer
def run_query_model(query: str, user_id: int, is_query_legal):
    if user_id not in user_sessions:
        return "No file uploaded yet. Please /start to upload a document first."
    
    user_data = user_sessions[user_id]
    answer, user_data["conversation_history"] = query_model(user_data["bm25"], user_data["vector_store"], query, is_query_legal, user_data["conversation_history"])
    return answer


# Help command
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("""Available Commands :- 
    /help - To get available commands.
    /start - To initialize the chat.
    /end - To end the chat and delete uploaded files.""")



# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_name = update.effective_user.username

    user_logger = get_user_logger(user_id, user_name)
    user_logger.info(f"User {update.effective_user.id} - {update.effective_user.username} started the bot.")

    print(f"User {update.effective_user.id} - {update.effective_user.username} started the bot.")
    # logger.info(f"User {update.effective_user.id} - {update.effective_user.username} started the bot.")
    await update.message.reply_text(f"Welcome {update.effective_user.first_name}! Please upload a document to start processing.")



# Handle text queries
async def handle_text_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_name = update.effective_user.username
    query_text = update.message.text.lower().strip()  # Convert to lowercase for case-insensitive matching

    user_logger = get_user_logger(user_id, user_name)
    user_logger.info(f"User {user_id} - {user_name} queried: {query_text}")
    # logger.info(f"User {user_id} - {user_name} queried: {query_text}")

    # Check if the message is a greeting
    greetings = {"hi", "hello", "hey", "hola", "namaste", "greetings"}
    if query_text in greetings:
        await update.message.reply_text("""Available Commands :- 
        /help - To get available commands.
        /start - To initialize the chat.
        /end - To end the chat and delete uploaded files.""")
        return

    # Process as a normal query
    # print(f"User {user_id} Query: {query_text}")
    try:
        # Show "typing..." animation
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        # is_query_legal = classify_query(query_text, user_sessions[user_id]["conversation_history"])
        res = process_query(query_text, user_sessions[user_id]["conversation_history"])
        is_query_legal = res['classification'] == 'legal'
        print(f"{query_text} :  {res}")
        print(f"\n Classification:  {is_query_legal}")

        if is_query_legal == True:
            # Send a temporary message to indicate processing
            waiting_message = await update.message.reply_text("🤓Generating answer...")
        else:
            waiting_message = await update.message.reply_text("✨Loading...")

        answer = run_query_model(query_text, user_id, is_query_legal)
        await waiting_message.edit_text(answer)
    except Exception as e:
        # logger.error(f"Error processing query from user {user_id}: {str(e)}")
        print(e)
        traceback.print_exc()
        await update.message.reply_text("Sorry, an error occurred while processing your request.")


# Function to end chat for a specific user
async def end_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    # Check if the user has an active session
    if user_id in user_sessions:
        # Delete all uploaded files for this user
        files = glob.glob(os.path.join(user_sessions[user_id]["upload_folder"], "*"))
        for file in files:
            os.remove(file)
        
        # Remove the user session
        del user_sessions[user_id]

        await update.message.reply_text("Chat session ended. All uploaded files have been deleted.")
    else:
        await update.message.reply_text("No active session found. Please upload a document first.")




def main():
    # Bot token from BotFather
    # token = os.getenv('legalsathi_api')
    token = os.getenv('botfather_api')
    
    # Set up the Application
    application = ApplicationBuilder().token(token).build()

    print("Setup Completed. Bot is ready.")

    # Add handlers
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))  # Handle file uploads
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_query))  # Handle normal text
    application.add_handler(CommandHandler("end", end_chat))  # Handles /end command

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
