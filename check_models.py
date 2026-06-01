# Run this from your project root to list available embedding models
from dotenv import load_dotenv
load_dotenv()
import os
import google as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available embedding models:")
for m in genai.list_models():
    if "embed" in m.name.lower() or "embedding" in m.supported_generation_methods.__str__().lower():
        print(f"  {m.name}  —  {m.supported_generation_methods}")