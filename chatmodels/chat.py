## not free to use
from dotenv import load_dotenv

load_dotenv()

import os
from langchain.chat_models import init_chat_model



model = init_chat_model("gpt-5.2")


response = model.invoke("Hello")

print(response.content)