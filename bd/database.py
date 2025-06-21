import pymongo
from config.config import DB_MONGO

def get_mongo_client():
    """Devuelve el cliente de conexión a MongoDB."""
    return pymongo.MongoClient(DB_MONGO["uri"])

def get_mongo_db():
    """Devuelve la base de datos MongoDB para EcommerTenants."""
    client = get_mongo_client()
    return client[DB_MONGO["db_name"]]