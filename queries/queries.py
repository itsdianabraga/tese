import csv
import json

with open("C:/Users/diana/PycharmProjects/thesis/keywords_definitions/synonyms.json") as file:
    data1 = json.load(file)

with open("C:/Users/diana/PycharmProjects/thesis/keywords_definitions/kids.json") as file:
    data2 = json.load(file)

print(data1)
print(data1.values())

queries = []
query_id = 1

for i in data1.values():
    for j in i:
        query = f'"{j}" ("children" OR "child" OR "kid" OR "youngster" OR "minor" OR "baby" OR "teenager" OR "infant" OR "adolescent" OR "juvenile" OR "kiddie" OR "youth" OR "pediatrics" OR "neonatal") -is:retweet -is:quote -is:reply'
        queries.append((query_id, query))
        query_id += 1

with open("queries.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["queryID", "query"])
    for query in queries:
        writer.writerow(query)
