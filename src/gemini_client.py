import os
import re
from dotenv import load_dotenv
from google import genai
from config import MODEL_NAME
from data_loader import load_images_as_base64

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini_order(video_id, shuffled=False):
    images, indices = load_images_as_base64(video_id, shuffled)
    contents = []
    for img in images:
        contents.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img
            }
        })
    contents.append({
        "text": "These are 8 frames from a video labeled 0 to 7 in the order shown. Arrange them in the correct temporal order they would appear in the original video. Reply with ONLY a Python list like [3,0,1,2,4,5,6,7]. No explanation."
    })
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents
    )
    return response.text, indices

def parse_order(response_text):
    match = re.search(r'\[[\d,\s]+\]', response_text)
    if match:
        return eval(match.group())
    return None