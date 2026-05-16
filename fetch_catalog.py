"""Download the SHL product catalog into catalog_cache.json."""
import json
import urllib.request

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTFILE = "catalog_cache.json"

with urllib.request.urlopen(CATALOG_URL, timeout=30) as response:
    raw = response.read().decode("utf-8", errors="replace")

# The provided SHL JSON currently contains at least one unescaped control
# character inside a string. strict=False reads it, and json.dump writes a
# clean standards-compliant cache for the app and deployment.
data = json.loads(raw, strict=False)

with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Downloaded {len(data)} SHL products to {OUTFILE}")
