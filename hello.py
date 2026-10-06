from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=1,
    messages=[{"role": "user", "content": "给沈杨（男）李雨伦（女）的男孩儿取名，并解释为什么这么取"}],
)
print(resp.choices[0].message.content)