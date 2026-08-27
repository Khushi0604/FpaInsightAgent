from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

print("Key being used:", os.getenv('GROQ_API_KEY'))
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "say hello"}]
)
print(response.choices[0].message.content)