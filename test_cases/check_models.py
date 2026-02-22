import os
import google.generativeai as genai

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzA8Uf_xfghL2ypS6EhuCHIy1eI2zzdt8'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

print("Available Gemini models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")
