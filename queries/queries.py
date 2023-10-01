import csv
import json

with open("C:/Users/diana/PycharmProjects/thesis/keywords_definitions/synonyms.json") as file:
    data1 = json.load(file)

print(data1.values())

queries = []
query_id = 1

for key, i in data1.items():
    main_topic= key
    for j in i:
        query = f'"{j}" ("children" OR "child" OR "kid" OR "youngster" OR "minor" OR "teenager" OR "teen" OR "infant" OR "adolescent" OR "juvenile" OR "youth" OR "pediatrics" OR "neonatal") -is:retweet -is:quote -is:reply'
        queries.append({"queryID": query_id,"query": query,"main_topic":main_topic})
        query_id += 1

with open("queries.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["queryID", "query", "main_topic"])
    for query in queries:
        writer.writerow(query)
