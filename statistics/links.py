import json
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Load the JSON file
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_tweets_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


# Dictionary to store the links for each main_topic
main_topic_links = {}

# Iterate through the data
for tweet_data in data.values():
    main_topic = tweet_data['query']['topic']
    links = tweet_data['metadata']['links']
    pontuation = tweet_data['metadata'].get('pontuation', 0)
    verified_user = tweet_data['user'].get('verified', False)

    # Check if links are available for the main_topic
    if links is not None:
        # Filter links with research or study-related keywords using regular expressions
        research_links = [link for link in links if link['title'] is not None and re.search(r'research|study|discoveries|studies|breakthrough|advancement|innovation|investigation|analyses|observations|developments|finding|scientific|experimental|surveys|conference|convention|seminar|webinar', link['title'], re.IGNORECASE)]

        # Add the pontuation to each link dictionary
        for link in research_links:
            link['pontuation'] = pontuation

        # Store the links in the main_topic_links dictionary
        main_topic_links.setdefault(main_topic, [])
        main_topic_links[main_topic].extend(research_links)

# Sort the links for each main_topic based on the link's pontuation in descending order
for main_topic, links in main_topic_links.items():
    links.sort(key=lambda link: link.get('pontuation', 0), reverse=True)

# Create a list to store the data
data = []

# Iterate over the main_topic_links dictionary
for main_topic, links in main_topic_links.items():
    if links != []:
        for link in links:
            title = link.get('title', 'N/A')
            description = link.get('description','N/A')
            title_pt = link.get('title_pt', 'N/A')
            description_pt = link.get('description_pt', 'N/A')
            url = link['url']
            pontuation = link.get('pontuation', 0)
            verified = link.get('user', {}).get('verified', False)

            # Create a dictionary for each link
            link_data = {
                "main_topic": main_topic,
                "url": url,
                "title": title,
                "title_pt" : title_pt,
                "description": description,
                "description_pt":description_pt,
                "pontuation": pontuation,
                "verified": verified
            }

            # Append the link data to the list
            data.append(link_data)

# Write the data to a JSON file
with open('output.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)