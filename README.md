Steps to run Telegram bot locally:

1. Clone the Repository
Open a terminal and run:
  git clone https://github.com/smanav00/JustiFi_bot
  cd JustiFi_bot

2. Create a Python Virtual Environment (Optional but Recommended) 
  python -m venv venv
  source venv/bin/activate   # On Linux/Mac
  venv\\Scripts\\activate    # On Windows

3. Install Dependencies
Install the required packages using:
			pip install -r requirements.txt

4. Replace the Telegram Bot API Token
You need to create your own Telegram bot using BotFather:
  1.	Open Telegram and search for @BotFather.
  2.	Use /newbot command to create a new bot.
  3.	Set a name and username.
  4.	BotFather will give you an API token like: "123456789:ABCdefGhIJKlmNoPQRstuVWxyZ"


5.	Replace the token in your code. Look for a line in the bot script like:
  TELEGRAM_BOT_TOKEN = "YOUR_BOTFATHER_API_KEY"
Replace "YOUR_BOTFATHER_API_KEY" with the token you got from BotFather.
      
6. Run the Bot
  Make sure your local Ollama or TogetherAI API is set up (depending on your inference choice), then start the bot using:
		python telegram_bot_multiuser.py

7.  Use the telegram bot
