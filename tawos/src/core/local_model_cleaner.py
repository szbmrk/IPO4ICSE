import asyncio
import pandas as pd
import json
import re
from config_loader import config

import aiohttp


def __extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(0))


async def __classify_single(session: aiohttp.ClientSession, title: str, desc: str):
    prompt = f"""
You are a classifier. Output ONLY valid JSON. 
The JSON must be: {{"point": <integer from 0 to 100>}}

Title: "{title}"
Description: "{desc}"
Respond only with JSON.
"""
    payload = {"prompt": prompt, "n_predict": 40, "temperature": 0}

    try:
        async with session.post(config.LOCAL_MODEL_URL, json=payload) as resp:
            resp_json = await resp.json()
            raw = resp_json.get("content", "")
            data = __extract_json(raw)
            return data["point"] if data else -1
    except Exception:
        return -1


async def __classify_batch(rows):
    async with aiohttp.ClientSession() as session:
        tasks = [__classify_single(session, title, desc) for title, desc in rows]
        return await asyncio.gather(*tasks)


async def classify_by_local_model(df: pd.DataFrame):
    rows = list(df[["Title", "Description"]].itertuples(index=False, name=None))

    results = []

    for i in range(0, len(rows), config.LOCAL_MODEL_BATCH_SIZE):
        batch = rows[i : i + config.LOCAL_MODEL_BATCH_SIZE]

        batch_results = await __classify_batch(batch)
        results.extend(batch_results)

    return results
