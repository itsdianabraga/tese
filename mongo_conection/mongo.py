from pymongo import MongoClient
import json

doc =open('C:/Users/diana/PycharmProjects/thesis/statistics/statistics_each_new.json', encoding='utf-8')
data = json.load(doc)

# Connect to the MongoDB Atlas cluster
client = MongoClient("mongodb+srv://diana-braga:test@cluster77.gjfoaqm.mongodb.net/?retryWrites=true&w=majority")

# Access the database
db = client['socialistening']

# Access the collection
#collection = db['links']
collection = db['statistics']

result = collection.delete_many({})

# Delete all documents in the collection
result_insert = collection.insert_many(data)

collection = db['statistics_general']

doc =open('C:/Users/diana/PycharmProjects/thesis/statistics/statistics_new.json', encoding='utf-8')
data = json.load(doc)

result = collection.delete_many({})

# Delete all documents in the collection
result_insert = collection.insert_many(data)

collection = db['general_info']

doc =open('C:/Users/diana/PycharmProjects/thesis/statistics/general_info.json', encoding='utf-8')
data = json.load(doc)

result = collection.delete_many({})

# Delete all documents in the collection
result_insert = collection.insert_many(data)

# Close the MongoDB connection
client.close()

