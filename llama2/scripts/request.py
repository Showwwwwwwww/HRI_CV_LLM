import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", 
    api_key = "echo in the moon"
)

completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
    {"role": "Shuo Chen", "content": "Who is talking with you?"}
]
)
response = completion.choices[0].message
print(response)
response = completion.choices[0].message.content
print(response)