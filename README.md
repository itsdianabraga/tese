# Tweet Processing and Analysis

## Overview

This project involves a series of scripts designed to extract, pre-process, and analyze tweets using the Twitter API. The scripts must be executed in a specific order to ensure smooth data extraction and processing.

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
python tweets_extraction.py
