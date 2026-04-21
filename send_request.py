import json
import openai

client = openai.Client(base_url="http://127.0.0.1:30010/v1", api_key="None")

response = client.chat.completions.create(
    model="./Qwen3-235B",
    messages=[
        {
            "role": "user",
            "content": "Who are you?",
        },
    ],
    temperature=0.5,
    max_tokens=500,
)
print(json.loads(response.model_dump_json())['choices'][0]['message']['content'])