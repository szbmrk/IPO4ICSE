from config_loader import config
from core.base_classifier import BaseClassifier, extract_json


class GeminiClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(
            model_name=config.GEMINI_MODEL,
            batch_size=config.GEMINI_BATCH_SIZE,
            temp_file="temp_results_gemini.csv",
        )

    async def _get_model_name(self):
        return self.model_name.replace("/", "_")

    async def _classify_single(self, session, title, desc):
        prompt = f"""
You are a classifier. Output ONLY valid JSON. 
The JSON must be: {{"p": <integer from 0 to 100>}}
Never write explanations or comments.

Your task is to give points to issues based on validity. 0 is spam, 100 is legit.

Issue:
Title: "{title}"
Description: "{desc}"

Respond only with JSON.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        params = {"key": config.GEMINI_API_KEY}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.GEMINI_TEMP,
                "maxOutputTokens": config.GEMINI_N_PREDICT,
            },
        }

        async with session.post(url, params=params, json=payload) as resp:
            try:
                data = await resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                return extract_json(raw)
            except Exception:
                return None
