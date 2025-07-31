from datetime import datetime

class MetadataTracker:
    def __init__(self, db, market):
        self.collection = db['metadata']
        self.market = market

    def update_metadata(self, symbol, interval, provider, timestamp=None):
        timestamp = timestamp or datetime.utcnow().isoformat()
        self.collection.update_one(
            {
                'symbol': symbol,
                'interval': interval,
                'provider': provider,
                'market': self.market
            },
            {
                '$set': {'last_updated': timestamp}
            },
            upsert=True
        )
