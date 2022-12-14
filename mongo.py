from pymongo import MongoClient
import uuid
import datetime
import pprint
from bson.objectid import ObjectId

# Connect to the server
client = MongoClient('mongodb+srv://aiteacher:1234@aiteacher.2urehvj.mongodb.net/?retryWrites=true&w=majority')
# Connect to the database
mydb = client['aiteacher']
# Connect to the Collection
coll = mydb['data']

# Collection에 저장될 metadata
document = {
    "images": { "cat": ["cat1.jpg"], "dog": ["dog1.jpg"] }, 
    "trained_model": "model_best_epoch.pt",
    "date": datetime.datetime.utcnow()
}

# Collection에 document 저장과 그에 해당하는 ObjectId 변수 저장
post_id = coll.insert_one(document).inserted_id

for x in coll.find():
    pprint.pprint(x)

coll.delete_one(({"_id": ObjectId(post_id)}))
for x in coll.find():
    pprint.pprint(x)