import numpy as np
from numpy.linalg import norm
import  os


def feature_compare(feature1, feature2, threshold=0.5):
    """
    Method 1:
        # Euclidean distance to compare with threshold
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False
    """
    # Method 2 - Cosine Similarity
    cosine = np.dot(feature1, feature2.T) / (norm(feature1) * norm(feature2))
    if cosine > 0.5:
        return True
    return False

def generate_conversation_prompt(com_face,dp_prompt=False):
    """
    Generates a prompt to continue a conversation with a newly recognized person,
    including their name, age, and gender.

    Parameters:
    - com_face (dict): A dictionary containing 'user_name', 'age', and 'sex' of the person.

    Returns:
    - str: A prompt for a large language model to continue the conversation.
    """
    # Extracting user information
    user_name = com_face['user_name']
    age =  com_face["age"]
    gender = com_face["sex"]

    if user_name == "ShuoChen":
        prompt = (
            f"[Here is the information for the poeple who interacting with you "
            f"{user_name} has join in the conversation, he is 21 years old and he is a Male. "
            f" He is a studnt at Monash University and studying computer science.")
        return prompt

    # Gender-neutral pronoun handling
    if gender == 'M':
        pronoun = 'he'
    else:
        pronoun = 'she'

    # Creating a welcoming and inclusive prompt
    prompt = (
        f"[Here is the information for Bob please do not response this prompt and continue your conversation "
        f"{user_name} has join in the conversation, he is {age} years old and identify as {gender}. "
        f"You are here to continue your engaging discussion and ensure it's inclusive and respectful. "
        f"{pronoun.capitalize()} has just joined us, and I'm looking forward to integrating "
        f"{pronoun}'s perspectives into our conversation. "
        f"Let's continue discussing with {user_name} in a way that's smooth and enjoyable for everyone involved."
        f"In order to do it, "
        f"You can initiate a more anthropomorphic conversation by asking questions of the current "
        f"use with the person name. You need to remember the all information for {user_name} with {age} and {gender} to "
        f"provied the better quliaty for the conversation.]"
    )
    return prompt
