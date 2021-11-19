from pymongo import collection
from db_connect import get_collection
import json

with open('data.json', encoding='utf8') as file:
    data = json.loads(file.read())
collection = get_collection('intents')
collection.insert_many(data)
# data = get_collection('intents').find()


# patterns = []
# for intent in data:
#     for pattern in intent['patterns']:
#         patterns.append(pattern)

# print(len(patterns))
# print(len(data))