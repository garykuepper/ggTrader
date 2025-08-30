# python
import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_crypto_symbols(num_currencies=250):
    url = f"https://finance.yahoo.com/crypto?offset=0&count={num_currencies}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    # Try to parse with pandas first (convenient if the table is well-formed)
    try:
        tables = pd.read_html(resp.text)
        if tables:
            df = tables[0].copy()
            # Ensure there's a Symbol column (sometimes column names differ)
            if "Symbol" in df.columns:
                return df
    except Exception:
        # fall through to BeautifulSoup fallback
        pass

    # Fallback: use BeautifulSoup to find the table and build DataFrame
    soup = BeautifulSoup(resp.text, "lxml")
    # Yahoo Finance crypto table rows are typically in <table> or in divs; try both
    table = soup.find("table")
    if table:
        try:
            df = pd.read_html(str(table))[0]
            return df
        except Exception:
            pass

    # If no <table>, try parsing rows from the page structure
    # Look for the rows that contain symbols (commonly in <a> tags with 'data-symbol' or a cell labeled Symbol)
    symbols = []
    rows = soup.select("tbody tr")
    for r in rows:
        # attempt to find the symbol in the first <a> or first <td>
        a = r.find("a")
        if a and a.text:
            symbols.append(a.text.strip())
        else:
            tds = r.find_all("td")
            if tds:
                symbols.append(tds[0].get_text(strip=True))

    if symbols:
        df = pd.DataFrame({"Symbol": symbols})
        return df

    # If everything failed, raise a clear error
    raise RuntimeError("Failed to parse cryptocurrency table from Yahoo Finance page.")

if __name__ == "__main__":
    num_currencies = 250
    try:
        df = fetch_crypto_symbols(num_currencies=num_currencies)
        symbols_yf = df["Symbol"].tolist() if "Symbol" in df.columns else df.iloc[:, 0].tolist()
        print(symbols_yf[:15])
        print(df.head(5))
    except Exception as e:
        print("Error fetching/parsing crypto symbols:", str(e))