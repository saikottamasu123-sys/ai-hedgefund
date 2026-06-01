# test_rag.py  — create this in your project root, run once, then delete
from dotenv import load_dotenv
load_dotenv()

from tools.financial_data import fetch_edgar_filing_rag

result = fetch_edgar_filing_rag("AAPL")

if result is None:
    print("FAILED — returned None")
else:
    print(f"Form:        {result['form']}")
    print(f"Filed:       {result['filing_date']}")
    print(f"Accession:   {result['accession_no']}")
    print(f"Text length: {len(result['text'])} chars")
    print()
    print("=== RETRIEVED TEXT ===")
    print(result['text'])