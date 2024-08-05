import openai as opai
import os
import json
import re
import requests

class llm:

    def __init__(self, path = './database/conversations'):
        self.client = opai.OpenAI(
            #base_url="http://localhost:8080/v1", 127.0.0.1
            base_url="http://127.0.0.2:8080/v1",
            api_key = "vl4ai"
        )
        self.path = path
        if not os.path.exists(self.path):
            print(f'Path: {self.path} not exit')
            os.makedirs(self.path)
        self.database_person = []
        self.conversations = self.initialize_conversation_history()
        self.last_person = None

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
                self.database_person.append(person)
                conversations[person] = file_info
                print(f'Person: {person} conversation loaded')
        return conversations

    def summarzie_conversation(self,person):
        """
        When the person leave the conversation, we need to summarize the conversation and start the new conversation?
        :param person:
        :return:
        """
        info = {}
        info['role'] = 'system'
        info['content'] = {
            'Summarize the conversation with the person and start the new conversation with the person'
        }
        pass

    def initialize_llm_talk(self, person = None, age = None, gender = None):
        if self.last_person == person:
            return None
        print(f'self.last_person: {self.last_person}, person: {person}')
        self.last_person = person  # new Person or back to the conversation
        if gender is None:
            gender = 'unknown'
        else:
            if age > 20:
                gender = 'Male' if gender == 'M' else 'Female'
            else:
                gender = 'Boy' if gender == 'M' else 'Girl'

        if  age is None:
            age = 'unknown'
        elif age < 13:
            age = 'child'
        elif age < 20:
            age =  'teenager'
        elif age < 60:
            age = 'adult'
        elif age >= 60:
            age = 'senior'

        if person not in self.conversations.keys():
            if 'Person' in person: # Person is unknown
                self.conversations[person] = [
                    {"role": "system",
                    "content":
                        f"Ginny, as a friendly and efficient social robot, your primary role is to engage in polite and engaging conversations."
                        f"When you meet someone for the first time, use the age group, gender, and available gene data to craft a personalized greeting." 
                        f"compliment them based on their age group and gender in a respectful manner, and kindly introduce yourself and ask for their name. "
                        f"This person looks like a {age} and {gender}, you need to correct with this person."
                        f"This is the example: Say something as complement considering its gender and age group together to start your conversation, eg I can see a cute small girl, correct? Or it seems a handsome young male in front of me, am I right. "
                        f"Then introduce yourself ask her/his name eg “My name is Ginny. What is your name?"
                        f"Notice you cannot use male or female or woman or adult in your response, this is rude to communicate with people with it"
                        f"Your response should not same sa the example, just use this as a reference and follow this structure "
                        f"reflecting the information provided during the interaction without making assumptions or faking details."
                        f"Your responses should be natural, friendly, and no more than 35 words. "
                        },
                ]
            else:
                self.conversations[person] = [
                    {"role": "system",
                    "content":
                        f"Ginny, as a friendly and efficient social robot, your primary role is to engage in polite and engaging conversations."
                        f"When you meet someone for the first time, use the age group, gender, and available gene data to craft a personalized greeting." 
                        f"This person name is {person}, and this person is {age} and {gender}."
                        f"you need to introduce yourself and kindly start the conversation with this person"
                        f"You want to know more about {person}, and use these information to optimized the conversation"
                        f"Your response can be dynamic but need to follow this structure. "
                        f"reflecting the information provided during the interaction without making assumptions or faking details."
                        f"Your responses should be natural, friendly, and no more than 35 words. "
                        f"Your all response do not have to use question as finish."
                        f"Ask question if necessary, otherwise use more natural communication flow."
                        },
                ]
            print(f'Person {person} initialized')
        else:
            info = {}
            info['role'] = 'system'
            info['content'] = (
                f"This person has returned to the conversation. Please continue your role as this person's friend and continue the conversation. "
                f"You need to welcome this person back to the conversation with this person's name and use a sentence of less than 30 words to resume the conversation based on the conversation history. "
                f"Ensure this person feel comfortable and the conversation flows naturally."
            )
            self.conversations[person].append(info)

        print(f'initialize the conversation with {person}')
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversations[person]
        )
        print(f'First conversation finished?')
        response = completion.choices[0].message.content
        print(completion.choices[0].message)
        # Check the length of the response
        while not self.evaluate_response_length(response):
            print(f' The response is too long, the response is: {response}')
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversations[person]
            )
            response = completion.choices[0].message.content
            #response = completion.choices[0].message.content
        # If the error happened, we need to do some process to fix the error
        response = self.process_response(response)
        self.conversations[person].append({
            'role': 'assistant',
            'content': response
        })
        return response

    def talkTollm(self,person,message):
        """
        Receive the message from person and talk with llm
        return: The response from the llm
        """
        # If the Person changed or just join

        print(f'Talk to LLM Start, the person is: {person}, the message is: {message}')
        info = {}
        info['role'] = 'user'
        info['content'] = message
        self.conversations[person].append(info)
        #print(f'Initialize the conversation with {person}, the conversation is: {self.conversations[person]}')
        #print(f'Save the conversation history for {person}.')
        print('Waiting for the response from LLM')
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversations[person]
            )
        response = completion.choices[0].message.content
        while not self.evaluate_response_length(response):
            print(f' The response is too long, the response is: {response}')
            info = {}
            info['role'] = 'system'
            info['content'] = (
                f"{person} is talking with you, and if you feel any confused about what his talking, you need to ask him again. "
                f"Besides, you need to ensure your response is start as the perspect from a friend and all your response have to "
                f"less than 30 words. "
            )
            self.conversations[person].append(info)
            # Send the response to the llm again until get a response with less than 30 words
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversations[person]
            )
            response = completion.choices[0].message.content
        print(completion.choices[0].message)
        #print(f'Complete the conversation')
        #print(completion.choices[0].message)
        response = self.process_response(response)
        self.conversations[person].append({
            'role': 'assistant',
            'content': response
        })
        return response
    
    def process_response(self, response):
        #print(f'The response before process is: {response}')
        pattern = r'^USER:\s*(.*?)s$'
        response = re.sub(pattern, r'\1', response)
        response = re.sub(r'</s>', '', response)
        cleaned_text = re.sub(r'(USER:|ASSISTANT:)', '', response)
        cleaned_text = re.sub(r'Assistant:', '', cleaned_text)
        response = cleaned_text.strip()
        #print(f'The response after process is: {response}')
        return response
    def save_conversations(self):
        print("Saving conversations...")
        for person, file_info in self.conversations.items():
            if person in self.database_person:
                file_path = os.path.join(self.path, f'{person}.json')
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(file_info, file, ensure_ascii=False, indent=4)
                print(f'People {person} conversation stored')
            else:
                print(f'People {person} conversation skipped')

    def evaluate_response_length(self,response):
        """
        Evaluate the response length
        """
        # Split the string into words
        words = response.split()
        # Count the number of words
        word_count = len(words)
        # Return False if word count is more than 30, else True
        res = (word_count <= 35)
        print(f' the Work count is: {word_count}, the response is: {response}')
        return res

    def __del__(self):
        """
        Save All conversation when delete 
        """
        print("Destructor called, saving all conversations.")
        self.save_conversations()

if __name__ == '__main__':
    agent = llm()
    agent.last_person = 'None'
    x = 0 
    while x < 7:
        agent.initialize_llm_talk('unknownperson',13,'F')
        user_response = input()
        response = agent.talkTollm('unknownperson',user_response)
        #print(response)
        x  += 1
    #agent.save_conversations()
    #del agent