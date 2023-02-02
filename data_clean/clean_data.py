#!/usr/bin/env python3
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time


l=links.get("metadata")

if l.get("links") is not None:
    for i in l.get("links"):
        title=i.get("title")
        description=i.get("description")
        url=i.get("expanded_url")
        images=i.get("images")
        im=[]
        for k in images:
            im.append(k.get("url"))
        print(title)
        print(description)
        print(im)