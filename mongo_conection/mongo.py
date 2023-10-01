import pymongo
'''
# Assuming you have a MongoDB client instance
client = pymongo.MongoClient("mongodb://localhost:27017/")

# Select the database and collection
db = client["admin"]
collection = db["Tweets-April"]

# Execute the query
query = {"query.id": {"$in": [18, 114, 210, 306, 402,498,594,690,786,882,978,1074,1170,1266]}, "sentiment": {"$eq": "Neutral"}}
results_neutral = collection.find(query)

query = {"query.id": {"$in": [18, 114, 210, 306, 402,498,594,690,786,882,978,1074,1170,1266]}, "sentiment": {"$eq": "Negative"}}
results_negative = collection.find(query)

query = {"query.id": {"$in": [18, 114, 210, 306, 402,498,594,690,786,882,978,1074,1170,1266]}, "sentiment": {"$eq": "Positive"}}
results_positive = collection.find(query)


i=0
# Process the query results
for result in results_neutral:
    i=i+1

g=0
# Process the query results
for result in results_negative:
    g=g+1

h=0
# Process the query results
for result in results_positive:
    h=h+1

print("Neutros: ",i)
print("Negativos: ",g)
print("Positivos: ",h)
'''
'''
import pymongo
from pymongo import MongoClient
cluster=MongoClient("mongodb+srv://diana-braga:test@cluster77.gjfoaqm.mongodb.net/?retryWrites=true&w=majority")
print(pymongo.__version__)

db = cluster["socialistening"]
collection = db["tweets"]

new_student =[{
    "Hey Yall!  Healthy habits matter!  Brush your teeth at least twice a day!   #kidshealth #k12 #healthyhabits  #children #teachers #parents #cartoons #disney": {
        "id_tweet": "1651004171070369793",
        "query": {
            "id": 1,
            "query": "\"healthy habits\" \"children\" -is:retweet -is:quote -is:reply"
        },
        "result": "Hey Yall! Healthy habits matter! Brush your teeth at least twice a day! #kidshealth #k12 #healthyhabits #children #teachers #parents #cartoons #disney ",
        "result_pt": "Olá a todos! Hábitos saudáveis são importantes! Escove os dentes pelo menos duas vezes por dia!#kidshealth #k12 #healthyhabits #children #teachers #parents #cartoons #disney",
        "link_tweet": "https://t.co/9cfZgmigW6",
        "question": False,
        "sentiment": "Neutral",
        "user": {
            "username": "TashaFNBNtv",
            "name": "It's Tasha",
            "verified": False,
            "profile_photo": "https://pbs.twimg.com/profile_images/1628796752039116802/KmqhcyXY_normal.jpg"
        },
        "nlp_process": {
            "tokens": [
                "healthy",
                "habits",
                "matter",
                "brush",
                "teeth",
                "twice",
                "kidshealth",
                "healthyhabits",
                "children",
                "teachers",
                "parents",
                "cartoons",
                "disney"
            ],
            "lemmas": [
                "healthy",
                "habit",
                "matter",
                "brush",
                "teeth",
                "twice",
                "kidshealth",
                "healthyhabits",
                "child",
                "teacher",
                "parent",
                "cartoon",
                "disney"
            ],
            "stems": [
                "healthi",
                "habit",
                "matter",
                "brush",
                "teeth",
                "twice",
                "kidshealth",
                "healthyhabit",
                "child",
                "teacher",
                "parent",
                "cartoon",
                "disney"
            ]
        },
        "metadata": {
            "pontuation": 24,
            "pontuation_mentions": 1,
            "lang": "en",
            "location": None,
            "links": None,
            "media": [
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/FumLQNdXsAI7Cqn.jpg"
                }
            ],
            "hashtags": [
                "#kidshealth",
                "#k12",
                "#healthyhabits",
                "#children",
                "#teachers",
                "#parents",
                "#cartoons",
                "#disney"
            ],
            "mentions": None,
            "entities": None,
            "public_metrics": {
                "impression_count": 23,
                "like_count": 0,
                "quote_count": 0,
                "reply_count": 0,
                "retweet_count": 0
            }
        }
    }},{
    "Wrapped up a great meeting with reps from @HeartandStroke where we discussed issues related to children's health and nutrition.  We all know how important it is to build healthy habits early on in life.   So let's keep working together to give our kids the best start in life!": {
        "id_tweet": "1650982438082863105",
        "query": {
            "id": 1,
            "query": "\"healthy habits\" \"children\" -is:retweet -is:quote -is:reply"
        },
        "result": "Wrapped up a great meeting with reps from @HeartandStroke where we discussed issues related to children's health and nutrition. We all know how important it is to build healthy habits early on in life. So let's keep working together to give our kids the best start in life! ",
        "result_pt": "Encerraram uma ótima reunião com representantes de @HeArtAndstroke, onde discutimos questões relacionadas à saúde e nutrição das crianças. Todos sabemos o quanto é importante construir hábitos saudáveis no início da vida. Então, vamos continuar trabalhando juntos para dar aos nossos filhos o melhor começo na vida!",
        "link_tweet": "https://t.co/PjnThtOotx",
        "question": False,
        "sentiment": "Neutral",
        "user": {
            "username": "SoniaLiberal",
            "name": "Sonia Sidhu",
            "verified": False,
            "profile_photo": "https://pbs.twimg.com/profile_images/661252831337439232/EqxzNjvu_normal.jpg"
        },
        "nlp_process": {
            "tokens": [
                "wrapped",
                "great",
                "meeting",
                "reps",
                "discussed",
                "issues",
                "related",
                "children",
                "health",
                "nutrition",
                "know",
                "important",
                "build",
                "healthy",
                "habits",
                "early",
                "life",
                "working",
                "kids",
                "best",
                "start",
                "life"
            ],
            "lemmas": [
                "wrapped",
                "great",
                "meeting",
                "rep",
                "discussed",
                "issue",
                "related",
                "child",
                "health",
                "nutrition",
                "know",
                "important",
                "build",
                "healthy",
                "habit",
                "early",
                "life",
                "working",
                "kid",
                "best",
                "start",
                "life"
            ],
            "stems": [
                "wrap",
                "great",
                "meet",
                "rep",
                "discuss",
                "issu",
                "relat",
                "child",
                "health",
                "nutrit",
                "know",
                "import",
                "build",
                "healthi",
                "habit",
                "earli",
                "life",
                "work",
                "kid",
                "best",
                "start",
                "life"
            ]
        },
        "metadata": {
            "pontuation": 2756,
            "pontuation_mentions": 1,
            "lang": "en",
            "location": "Canada",
            "links": None,
            "media": [
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/Ful3xnFXwAEm5e7.jpg"
                },
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/Ful3xnGWAAEx-ih.jpg"
                },
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/Ful3xnHXgAAzAIe.jpg"
                },
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/Ful3xnGX0AAxilz.jpg"
                }
            ],
            "hashtags": None,
            "mentions": None,
            "entities": None,
            "public_metrics": {
                "impression_count": 2723,
                "like_count": 27,
                "quote_count": 1,
                "reply_count": 0,
                "retweet_count": 4
            }
        }
    }}]

collection.insert_many(new_student)
'''

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

