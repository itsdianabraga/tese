import json
import re
import requests
import PyPDF2
from bs4 import BeautifulSoup

# Function to retrieve the first relevant image URL from a Google Image search
def get_google_photos_image(query):
    """
    Perform a Google image search and return the URL of the second image result.
    
    Args:
        query (str): The search query string.
        
    Returns:
        str: The URL of the second image result, or a default image URL if unsuccessful.
    """
    default_img_url = "https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/check_up_hero_2.jpg"
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")
        
        # Return the second image's URL if available
        if len(img_tags) >= 2:
            return img_tags[1].get("src", default_img_url)
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
        pass  # Return the default image URL in case of any request errors

    return default_img_url

# Function to extract site information, including the title and an image URL
def get_site_info(url, query):
    """
    Fetch the title and image URL from the provided URL using Open Graph tags or PDF metadata.
    
    Args:
        url (str): The target URL.
        query (str): The fallback query string used if the title cannot be fetched.
        
    Returns:
        tuple: A tuple containing the article title and image URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
        return query, None

    # Handle PDF files separately
    if url.endswith('.pdf'):
        pdf_image_url = "https://s.yimg.com/uu/api/res/1.2/YS6jQafc4MCmwFkddc3e2A--~B/Zmk9ZmlsbDtoPTE1MTg7dz0yNDcwO2FwcGlkPXl0YWNoeW9u/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2022-07/df642320-090d-11ed-babf-97960e1b79a0.cf.jpg"
        try:
            pdf_content = requests.get(url).content
            with open("temp.pdf", "wb") as pdf_file:
                pdf_file.write(pdf_content)

            with open("temp.pdf", "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                title = pdf_reader.metadata.title
            return title if title else query, pdf_image_url
        except Exception:
            return query, pdf_image_url

    # Parse HTML content to extract Open Graph metadata
    soup = BeautifulSoup(response.text, 'html.parser')
    og_title = soup.find('meta', property='og:title')
    og_image = soup.find('meta', property='og:image')

    title = og_title['content'] if og_title else query
    image_url = og_image['content'] if og_image else None

    return title, image_url

# Function to check if a link is active
def is_link_active(link):
    """
    Check if a link is active by sending a HEAD request.
    
    Args:
        link (str): The URL to check.
        
    Returns:
        bool: True if the link is active, False otherwise.
    """
    try:
        response = requests.head(link, timeout=10)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

# Load and process JSON data
with open('C:/Users/diana/PycharmProjects/thesis/data_clean/clean_june.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

research_links = []

# Iterate through JSON data and filter research-related links
for tweet_data in data.values():
    topic = tweet_data['query']['topic']
    links = tweet_data['metadata']['links']
    media = tweet_data['metadata']['media']
    pontuation = tweet_data['metadata'].get('pontuation')
    verified_user = tweet_data['user'].get('verified')

    if links:
        for link in links:
            title = link.get("title", " ")
            description = link.get("description", " ")
            site = link.get("url")
            quote = f"{title} {description}".lower()

            # Identify research-related content through keyword matching
            if re.search(r'research|article|study|discover|advancement|innovation|investigation|analysis|scientific|survey|conference|science|scientist|\.gov', quote) or \
               re.search(r'research|article|study|discover|advancement|innovation|investigation|analysis|scientific|survey|conference|science|scientist|\.gov', site):
                link["pontuation"] = pontuation
                link["topic"] = topic
                link["verified"] = verified_user

                # Attach the first photo URL from media if available
                if media:
                    for photo in media:
                        if photo['type'] == "photo":
                            link["photo"] = photo['url']

                research_links.append(link)

# Sort the research links by pontuation
sorted_data = sorted(research_links, key=lambda x: x["pontuation"], reverse=True)

# Prepare the final dataset
final_data = []

# Process each sorted link
for i, link in enumerate(sorted_data, start=1):
    url = link['url']

    if is_link_active(url):
        title = link.get('title', 'Not Available')
        topic = link.get('topic', 'N/A')
        title, first_image_url = get_site_info(url, title)

        # Use Google Images as a fallback if no image was found
        if not first_image_url:
            first_image_url = get_google_photos_image(topic)

        description = link.get('description', 'N/A')
        title_pt = link.get('title_pt', 'N/A')
        description_pt = link.get('description_pt', 'N/A')
        pontuation = link.get('pontuation', 0)
        verified = link.get('verified', False)

        # Append the processed link data to the final dataset
        final_data.append({
            "topic": topic,
            "url": url,
            "title": title,
            "title_pt": title_pt,
            "description": description,
            "description_pt": description_pt,
            "pontuation": pontuation,
            "verified": verified,
            "photo": first_image_url
        })

# Write the final dataset to a JSON file
with open('links.json', 'w', encoding='utf-8') as json_file:
    json.dump(final_data, json_file, indent=4, ensure_ascii=False)
