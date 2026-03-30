import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

response = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",  # free model via OpenRouter
    messages=[
        {"role": "user", "content": "What is AI in simple terms?"}
    ]
)

print(response.choices[0].message.content)