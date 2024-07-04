import openai

class llm:

    def __init__(self):
        self.client = openai.OpenAI(
            base_url="http://localhost:8080/v1", 
            api_key = "echo in the moon"
        )
        self.messageHistory = {}

    def initialize_messageHistory(self,path = None):
        """
        Load the Json file to extract all conversation history
        """
        pass 
    def talkTollm(self,people,message):
        """
        Receive the message from person
        """
        info = {}
        info['role'] = people
        info['content'] = message

        self.messageHistory[people].append(info)
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messageHistory[people]
            )
        response = completion.choices[0].message.content
        #print(response)
        return response
    