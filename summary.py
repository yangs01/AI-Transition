from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("spec.txt", "r", encoding="utf-8") as f:
    text = f.read()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.3,
    messages=[{"role": "user", "content": f"请总结以下文本的要点：\n\n{text}"}],
)
print(resp.choices[0].message.content)
