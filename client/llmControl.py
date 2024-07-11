import openai as opai
import os
import json



class llm:

    def __init__(self, path = './database/conversations'):
        self.client = opai.OpenAI(
            base_url="http://localhost:8080/v1", 
            api_key = "echo in the moon"
        )
        self.path = path
        if not os.path.exists(self.path):
            print(f'Path: {self.path} not exit')
            os.makedirs(self.path)
        self.conversations = self.initialize_conversation_history()

    def initialize_conversation_history(self):
        """
        Load the Json file to extract all conversation history, One Person, One Json 
        """
        conversations = {}
        for filename in os.listdir(self.path):
            if filename.endswith('.json'): 
                file_path = os.path.join(self.path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_info = json.load(file)
                person = filename[:-5]
                conversations[person] = file_info
                print(f'Person: {person} conversation loaded') 
        return conversations

    def talkTollm(self,person,message):
        """
        Receive the message from person and talk with llm
        return: The response from the llm
        """
        info = {}
        info['role'] = 'user'
        info['content'] = message
        print(f'person: {person}')
        print(f' keys: {self.conversations.keys()}')
        if person not in self.conversations.keys():
            self.conversations[person] = [
                        {"role": "ASSISTANT", 
                        "content": "You are a friend social Robot, now there is a User talking with you. At first you need to ask question to him to know the basic information about this person and them kindly talking with him. Whatever user says, you need to generate response.All your talking have to smaller than 30 words."},
                       ]

        self.conversations[person].append(info)
        print(self.conversations[person])
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversations[person]
            )
        response = completion.choices[0].message.content
        print(response)
        self.conversations[person].append({
            'role': 'ASSISTANT',
            'content': response
        })
        return response
    
    def save_conversations(self):
        # print("Saving conversations...")
        # print(self.conversations)
        for person, file_info in self.conversations.items():
            file_path = os.path.join(self.path, f'{person}.json')
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(file_info, file, ensure_ascii=False, indent=4)
            print(f'People {person} conversation stored')

    def __del__(self):
        """
        Save All conversation when delete 
        """
        print("Destructor called, saving all conversations.")
        self.save_conversations()

if __name__ == '__main__':
    agent = llm()
    x = 0 
    while x < 5:
        user_response = input()
        response = agent.talkTollm('Friend2',user_response)
        #print(response)
        x  += 1
    agent.save_conversations()
    del agent