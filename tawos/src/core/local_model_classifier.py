import aiohttp
import os

from config_loader import config
from core.base_classifier import BaseClassifier, extract_json


class LocalModelClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(
            model_name=config.LOCAL_MODEL_URL,
            batch_size=config.LOCAL_MODEL_BATCH_SIZE,
        )

    async def _get_model_name(self) -> str:
        url = self.model_name + "/v1/models"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                try:
                    data = await resp.json()
                    full_path = data["models"][0]["name"]
                    filename = os.path.basename(full_path)
                    name = os.path.splitext(filename)[0]
                    return name.replace(".gguf", "")
                except Exception:
                    return "unkown_model"

    async def _classify_single(self, session, title, desc) -> int | None:
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
        payload = {
            "prompt": prompt,
            "n_predict": config.LOCAL_MODEL_N_PREDICT,
            "temperature": config.LOCAL_MODEL_TEMP,
        }

        try:
            async with session.post(
                f"{self.model_name}/completions", json=payload
            ) as resp:
                resp_json = await resp.json()
                raw = resp_json.get("content", "")
                return extract_json(raw)
        except Exception:
            return None
