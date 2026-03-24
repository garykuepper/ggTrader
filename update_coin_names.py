import json
import os

SYMBOL_TO_NAME = {
    'AAVE': 'Aave',
    'ADA': 'Cardano',
    'AKT': 'Akash Network',
    'ALGO': 'Algorand',
    'ARB': 'Arbitrum',
    'ATOM': 'Cosmos',
    'AVAX': 'Avalanche',
    'BCH': 'Bitcoin Cash',
    'BONK': 'Bonk',
    'BTC': 'Bitcoin',
    'CFG': 'Centrifuge',
    'CRV': 'Curve DAO Token',
    'DOGE': 'Dogecoin',
    'DOT': 'Polkadot',
    'ENA': 'Ethena',
    'ETH': 'Ethereum',
    'EWT': 'Energy Web Token',
    'FET': 'Fetch.ai',
    'FIL': 'Filecoin',
    'FLR': 'Flare',
    'FTM': 'Fantom',
    'GALA': 'Gala',
    'GRT': 'The Graph',
    'ICP': 'Internet Computer',
    'IMX': 'Immutable',
    'INJ': 'Injective',
    'KAS': 'Kaspa',
    'KSM': 'Kusama',
    'LDO': 'Lido DAO',
    'LINK': 'Chainlink',
    'LTC': 'Litecoin',
    'LUNA': 'Terra',
    'MATIC': 'Polygon',
    'NEAR': 'Near Protocol',
    'OCEAN': 'Ocean Protocol',
    'ONDO': 'Ondo',
    'PEPE': 'Pepe',
    'QNT': 'Quant',
    'RENDER': 'Render Token',
    'SAND': 'The Sandbox',
    'SCRT': 'Secret',
    'SEI': 'Sei',
    'SGB': 'Songbird',
    'SHIB': 'Shiba Inu',
    'SOL': 'Solana',
    'SPX': 'SPX6900', # Assuming SPX6900 based on common crypto meme coins on Kraken
    'STX': 'Stacks',
    'SUI': 'Sui',
    'TAO': 'Bittensor',
    'TIA': 'Celestia',
    'TRX': 'TRON',
    'UNI': 'Uniswap',
    'WIF': 'dogwifhat',
    'XCN': 'Onyxcoin',
    'XLM': 'Stellar',
    'XMR': 'Monero',
    'XRP': 'XRP',
    'ZEC': 'Zcash'
}

data_dir = r"c:\Users\gkuep\PycharmProjects\ggTrader\data"

for filename in os.listdir(data_dir):
    if filename.endswith(".json") and filename != ".processed_dirs.json":
        filepath = os.path.join(data_dir, filename)
        print(f"Processing {filename}...")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                updated = False
                for item in data:
                    if isinstance(item, dict) and "symbol" in item:
                        symbol = item["symbol"]
                        if symbol in SYMBOL_TO_NAME:
                            item["name"] = SYMBOL_TO_NAME[symbol]
                            updated = True
                
                if updated:
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"  Updated {filename}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
