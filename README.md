# Tweet Analysis Project

## Overview

This project involves a series of scripts designed to extract, pre-process, and analyze tweets using the Twitter API. The scripts must be executed in a specific order to ensure smooth data extraction and processing. For convenience, you can use the `main.py` script to run all of the scripts in the correct order.

## Requirements

The project requires the following Python packages:

- `requests`
- `urllib3`
- `spacy`
- `langid`
- `googletrans`
- `textblob`
- `pandas`
- `nltk`
- `pysentimiento`
- `transformers`
- `torch`

For the exact versions, refer to `requirements.txt`.

## Installation

To set up the project environment, you need to install the required dependencies. You can do this using pip:

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. **Twitter API Key:**
   - Update the `bearer_token` in `tweets_extraction.py` with your Twitter API bearer token.

2. **Google Maps API Key:**
   - Update the `API_KEY` in `pre-processing.py` with your Google Maps API key.

3. **Queries File:**
   - Ensure `queries.json` contains the query parameters for extracting tweets.

4. **File Paths:**
   - Verify and update file paths in each script if necessary.

## Usage

To run the entire pipeline in the correct order, you can use the `main.py` script. This script will automatically execute `tweets_extraction.py`, `pre-processing.py`, `processing.py`, statistics python files and `mongo_db`.

**How to Run:**
```bash
python main.py
```

## Scripts

### 1. `tweets_extraction.py`

**Purpose:** This script connects to the Twitter API to extract tweets based on specific queries and keywords. It handles API rate limits and retries, processes the tweet data, and stores it in a structured format.

**Key Features:**
- **API Connection:** Utilizes bearer token for authentication and handles rate limiting with retry logic.
- **Tweet Processing:** Extracts tweet details, media attachments, user information, and metadata.
- **Rate Limiting:** Manages API rate limits to avoid hitting usage caps.
- **Data Storage:** Saves the extracted tweets to a JSON file.

**Usage:**
- Set your bearer token and other configurations.
- Ensure the queries file `queries.json` is correctly formatted and located at the specified path.
- Run the script to extract tweets and save them to `tweets_june.json`.

**How to Run:**
```bash
python ./tweets_data/tweets_extraction.py
```

### 2. `pre-processing.py`

**Purpose:**  
This script processes the extracted tweets to clean and enrich the data. It performs tasks such as translation, text cleaning, sentiment analysis, and natural language processing (NLP).

**Key Features:**
- **Translation:** Translates tweets to Portuguese if they are not already in that language.
- **Text Cleaning:** Removes URLs, mentions, hashtags, and performs text normalization.
- **NLP Processing:** Applies tokenization, lemmatization, and stemming.
- **Location Processing:** Retrieves and processes location information using the Google Maps API.

**Usage:**
- Update API_KEY with your Google Maps API key.
- Ensure the `tweets_june.json` file is available and correctly formatted.
- Run the script to process tweets and save the cleaned data to `clean_tweets_june.json`.

**How to Run:**
```bash
python ./data_clean/pre-processing.py
```

### 3. `processing.py`

**Purpose:**  
This script further processes the cleaned tweets by performing keyword matching, sentiment analysis, and text normalization. It also categorizes tweets based on specific topics and saves the final cleaned and categorized data.

**Key Features:**
- **Keyword Matching:** Identifies tweets related to predefined health and food topics using keyword matching.
- **Sentiment Analysis:** Uses BERT-based sentiment analysis for determining tweet sentiment.
- **Text Normalization:** Cleans and normalizes text, removes URLs, mentions, and other irrelevant content.
- **Final Filtering:** Categorizes tweets based on topics and cleans up unnecessary fields.

**Usage:**
- Ensure the `clean_tweets_june.json` file is available and correctly formatted.
- Run the script to process and categorize tweets, then save the results to `limpeza_june.json`.

**How to Run:**
```bash
python ./data_clean/processing.py
```

### 4. `each_category.py`

**Purpose:**
This script analyzes processed tweet data to generate statistics and insights by topic. It performs clustering of tweets based on similarity, aggregates engagement metrics, calculates sentiment percentages, and provides geographical distributions. It also identifies and ranks top tweets based on various criteria and saves the results to a JSON file.

**Key Features:**
- **Clustering:** Groups similar tweets using cosine similarity and vectorization.
- **Engagement Metrics:** Aggregates likes, retweets, impressions, and other metrics.
- **Sentiment Analysis:** Calculates and reports sentiment percentages.
- **Geographical Distribution:** Maps tweet locations and calculates geographical distribution percentages.
- **Top Results:** Ranks top tweets based on engagement and other criteria.

**Usage:**
- Ensure the processed tweets file processados_june.json is available and correctly formatted.
- Run the script to analyze data and save results to statistics_each_new.json.

**How to Run:**
```bash
python ./statistics/each_category.py
```
