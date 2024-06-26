import os
import time
import json

def dict_to_string(dictionary):
    return json.dumps(dictionary, ensure_ascii=False)

info = {
    "name": "Shuo Chen",
    "age": 22,
    "gender": "Male",
    "education": "Monash University",
    "major": "Computer Science",
    "scenarios": [
        {
            "scenario": "Education",
            "role": "Student",
            "tutor_profile": {
                "learning_level": "College Student",
                "current_knowledge": {
                    "topics_interested": ["Algorithms", "Data Structures", "Artificial Intelligence"],
                    "prior_knowledge": {
                        "Algorithms": "Basic understanding",
                        "Data Structures": "Intermediate",
                        "Artificial Intelligence": "Beginner"
                    }
                },
                "learning_goals": ["Deepen understanding of AI", "Master advanced algorithms", "Improve coding skills"]
            }
        }
    ]
}

prefix = 'This is information for the people who is asking your questions.  '
questions = [
    "I want to understand artificial intelligence better, can you help me?",
    "I'm having trouble learning algorithms, can you give me some advice?",
    "Can you explain advanced concepts in data structures?",
    "What exercises or resources do you recommend to improve my coding skills?",
    "I already know some basic algorithms, can you give me some more advanced examples?"
]

file_path = 'output/exchange_information/py_to_cpp.txt'
info = dict_to_string(info)
count = 0

while count < 5:
    content = prefix + info + ' He is asking a question: ' + questions[count]
    input("Press Enter to continue...")

    x = time.time()
    while True:
        # Check if the file exists and if it is empty
        if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
            with open(file_path, 'w') as file:
                file.write(content)
                break  # Exit the loop after writing the content
        elif not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write(content)
                break  # Exit the loop after writing the content
        else:
            # Optional: Sleep for a short time to avoid busy waiting
            time.sleep(1)

    count += 1
    # Print the time taken for the operation
    print(f'Time is: {time.time() - x}')
