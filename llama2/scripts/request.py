import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", 
    api_key = "echo in the moon"
)

# completion = client.chat.completions.create(
# model="gpt-3.5-turbo",
# messages=[
#     {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
#      {"role": "system", "content": "Now a new person join the conversation, you need to aks his basic information and return as the Json format"},
# ]
# )
# response = completion.choices[0].message
# print(response)
# response = completion.choices[0].message.content
# print(response)

messages=[
    {"role": "system", "content": "You are a friend social Robot, now there is a new person talking with you. At first you need to ask question to him to know the basic information about this person and them kindly talking with him. All your talking have to smaller than 30 words"},
    {"role": "new_person", "content": "Hello" }
]
for _ in range(5):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
    )
    response = completion.choices[0].message.content
    print(response)
    temp = {}
    temp["role"] = "system"
    temp["content"] = response
    messages.append(temp)
    user_response = input()
    user = {}
    user["role"] = "new_person"
    user["content"] = user_response
    messages.append(user)