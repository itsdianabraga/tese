import json
import requests
from bs4 import BeautifulSoup
import re

# Load the JSON data from the file
with open('C:/Users/diana/PycharmProjects/thesis/tweets_june.json', encoding='utf-8') as file:
    data = json.load(file)

# Extract the locations from the JSON data
locations = set()
for tweet_data in data.values():
    location = tweet_data.get('metadata', {}).get('location')
    if location:
        print(location)
        url = f"http://www.geonames.org/search.html?q={location}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        element = soup.select_one("#search > table > tr:nth-child(3) > td:nth-child(3) > a")
        if element:
            print(element.text)
        else:
            print("Element not found")
        locations.add(location)

# Count the number of different locations
num_locations = len(locations)

# Print the result
print(f"The number of different locations in the JSON file is: {num_locations}")


