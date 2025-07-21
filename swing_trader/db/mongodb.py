from pymongo import MongoClient

def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    return client["swing_trader_db"]

def get_collection(name):
    db = get_db()
    return db[name]
