from pymongo import MongoClient

client = MongoClient("mongodb+srv://grindandeat786:rhWrhhL0LowOobZn@cluster0.ldwjttn.mongodb.net/")

db = client.complete_database

collection_name = db["license_plates"]

second_collection_name = db["Criminal_Database"]

third_collection_name = db["Authorization"]