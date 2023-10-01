import requests
import spacy


def is_link_active(link):
    try:
        response = requests.head(link, timeout=10)  # Add a timeout of 10 seconds
        print(response)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

# Example links
links = [
    "https://www.sacbee.com/news/equity-lab/article276167561.html"]
'''
for link in links:
    if is_link_active(link):
        print(f"{link} is active")
    else:
        print(f"{link} is not active")
'''

import spacy

result_en = "I had a minor stroke"
nlp = spacy.load("en_core_web_sm")  # Load a pre-trained English language model
doc = nlp(result_en)
for token in doc:
    print(token.text)
    print(token.pos_)
    # Check if the token is "minor" and its part-of-speech tag is "VERB"
    if token.text == "minor" and token.pos_ == "VERB":
        print(token.pos_)
