import json
from collections import Counter
import random
import PyPDF2
import requests
from bs4 import BeautifulSoup

# Function to perform a Google search and extract the first image URL
def get_google_photos_image(query):
    # Create a Google search URL with the query
    img_url = "https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/check_up_hero_2.jpg"
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"

    # Set the user-agent to avoid bot detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    # Send an HTTP GET request to the Google search URL
    try:
        response = requests.get(search_url, headers=headers, timeout=10)  # Add a timeout of 10 seconds
    except (requests.ConnectionError, requests.Timeout):
        return img_url

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all image elements on the page
        img_tags = soup.find_all("img")

        # Check if there is a second image tag
        if len(img_tags) >= 2:
            # Get the second image tag (index 1)
            second_img = img_tags[1]

            # Extract the image source URL
            if "src" in second_img.attrs:
                img_url = second_img["src"]
                return img_url

    return img_url

def get_site_info(urlll,query):
    url = urlll

    article_title = query
    image_url = None

    try:
        response = requests.get(url, timeout=10)  # Add a timeout of 10 seconds
    except (requests.ConnectionError, requests.Timeout):
        return article_title,image_url

    if re.findall(r'\.pdf',url):
        image_pdf="https://s.yimg.com/uu/api/res/1.2/YS6jQafc4MCmwFkddc3e2A--~B/Zmk9ZmlsbDtoPTE1MTg7dz0yNDcwO2FwcGlkPXl0YWNoeW9u/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2022-07/df642320-090d-11ed-babf-97960e1b79a0.cf.jpg"
        image_url = image_pdf
        responses = requests.get(urlll)
        with open("sg-youth-mental-health-social-media-advisory.pdf", "wb") as pdf_file:
            pdf_file.write(responses.content)

        # Open the downloaded PDF file
        with open("sg-youth-mental-health-social-media-advisory.pdf", "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract the title from the PDF metadata
            title = pdf_reader.metadata

            article_title = title.title

        return article_title, image_url

    elif response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title from the Open Graph meta tags
        og_title = soup.find('meta', property='og:title')
        if og_title:
            article_title = og_title['content']

        # Extract the image URL from the Open Graph meta tags
        og_image = soup.find('meta', property='og:image')
        if og_image:
            image_url = og_image['content']

        return article_title, image_url

    else:
        search_url = f"https://www.google.com/search?q={query}&tbm=isch"

        article_title=query

        # Set the user-agent to avoid bot detection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

        # Send an HTTP GET request to the Google search URL
        response = requests.get(search_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all image elements on the page
            img_tags = soup.find_all("img")

            # Check if there is a second image tag
            if len(img_tags) >= 2:
                # Get the second image tag (index 1)
                second_img = img_tags[1]

                # Extract the image source URL
                if "src" in second_img.attrs:
                    img_url = second_img["src"]
                    return article_title, image_url

    return article_title, image_url


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Load the JSON file
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

import requests

def is_link_active(link):
    try:
        response = requests.head(link, timeout=10)  # Add a timeout of 10 seconds
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

# Dictionary to store the links for each main_topic
main_topic_links = {}
research_links=[]

# Iterate through the data
for tweet_data in data.values():
    topic = tweet_data['query']['topic']
    links = tweet_data['metadata']['links']
    media = tweet_data['metadata']['media']
    pontuation = tweet_data['metadata'].get('pontuation')
    verified_user = tweet_data['user'].get('verified')

    # Check if links are available for the main_topic
    if links is not None:
        for link in links:
            url = "N/A"
            titulo=link.get("title"," ")
            description=link.get("description"," ")
            site=link.get("url")
            if titulo is None:
                titulo=" "
            if description is None:
                description= " "
            quote= titulo + " " + description
            quote=quote.lower()
        # Filter links with research or study-related keywords using regular expressions
            if re.search(r'research|article|study|discover|studies|advancement|innovation|investigation|analys|observation|development|finding|scientific|experimental|survey|conference|convention|seminar|webinar|pubmed|scielo|scientist|government|science', quote):
                link["pontuation"]=pontuation
                link["topic"]=topic
                link["verified"]=verified_user
                if media is not None:
                    for photo in media:
                        if photo['type'] == "photo":
                            url = photo['url']
                    link["photo"] = url
                research_links.append(link)
            elif re.search(r'research|article|study|discover|studies|advancement|innovation|investigation|analys|observation|development|finding|scientific|experimental|survey|conference|convention|seminar|webinar|pubmed|scielo|\.gov|scientist|government|science|www\.cnn', site):
                link["pontuation"] = pontuation
                link["topic"] = topic
                link["verified"] = verified_user
                if media is not None:
                    for photo in media:
                        if photo['type'] == "photo":
                            url = photo['url']
                    link["photo"] = url
                research_links.append(link)

print(len(research_links))
sorted_data = sorted(research_links, key=lambda x: x["pontuation"], reverse=True)

# Create a list to store the data
data = []

j=0
# Iterate over the main_topic_links dictionary
for l in sorted_data:
    print(j)
    j=j+1
    url = l['url']
    if is_link_active(url):
        title = l.get('title', 'Not Available')
        topic = l.get('topic')
        if isinstance(topic,list):
            t=topic[1]
        else:
            t=topic
        if title is None:
            title='Not Available'
        try:
            title,first_image_url=get_site_info(url,title)
        except:
            first_image_url==None
            print("não deu")
        if first_image_url==None:
            first_image_url=get_google_photos_image(t)
        description = l.get('description','N/A')
        title_pt = l.get('titule_pt', 'N/A')
        description_pt = l.get('description_pt', 'N/A')
        pontuation = l.get('pontuation', 0)
        verified = l.get('verified')
        # Create a dictionary for each link
        link_data = {
            "topic": topic,
            "url": url,
            "title": title,
            "title_pt" : title_pt,
            "description": description,
            "description_pt":description_pt,
            "pontuation": pontuation,
            "verified": verified,
            "photo": first_image_url
        }
        # Append the link data to the list
        data.append(link_data)

# Write the data to a JSON file
with open('links.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)