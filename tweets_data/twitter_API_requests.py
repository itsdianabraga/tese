#!/usr/bin/env python3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import json

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<AAAAAAAAAAAAAAAAAAAAAFdCiQEAAAAASlcblnjSnYEbgMIzB0Kd4SxDm6c%3D1eLJSoSTYavew0Ye0ZGMmvvrfC8SOvycnxRZvjImZtQjbgXaeX>'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFdCiQEAAAAASlcblnjSnYEbgMIzB0Kd4SxDm6c%3D1eLJSoSTYavew0Ye0ZGMmvvrfC8SOvycnxRZvjImZtQjbgXaeX'

search_url = "https://api.twitter.com/2/tweets/search/recent"
search_url1 = "https://api.twitter.com/2/tweets/search/all?"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
q = "beauty"


#'tweet.fields': 'author_id',"next_token": "b26v89c19zqg8o3fpzemtikjdotigv4y65mcc0x3nul4t", "max_results":500, "user.fields":"verified"
def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r


def connect_to_endpoint(url, params, q):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    response = session.get(url, auth=bearer_oauth, params={'query': q})
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    q = "beauty"
    json_response = connect_to_endpoint(search_url1, query_params,q)
    print(json.dumps(json_response, indent=4, sort_keys=True))

query_params = {'query': q}

main()