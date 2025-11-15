from config_loader import config
from core.base_classifier import BaseClassifier, extract_json


class ClaudeClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(
            model_name=config.CLAUDE_MODEL,
            batch_size=config.CLAUDE_BATCH_SIZE,
            temp_file="temp_results_claude.csv",
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

        url = f"https://api.anthropic.com/v1/models/{self.model_name}/classify"
        headers = {"Authorization": f"Bearer {config.CLAUDE_API_KEY}"}
        payload = {
            "prompt": prompt,
            "temperature": config.CLAUDE_TEMP,
            "max_tokens": config.CLAUDE_N_PREDICT,
        }

        async with session.post(url, headers=headers, json=payload) as resp:
            try:
                data = await resp.json()
                raw = data["completion"]
                return extract_json(raw)
            except Exception:
                return None
