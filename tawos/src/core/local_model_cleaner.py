import asyncio
import pandas as pd
import json
import re
import aiohttp
import os

from core.log import get_logger
from config_loader import config

logger = get_logger("LocalModel")


def __extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(0))


async def __classify_single(session: aiohttp.ClientSession, title: str, desc: str):
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
        async with session.post(config.LOCAL_MODEL_URL, json=payload) as resp:
            resp_json = await resp.json()
            raw = resp_json.get("content", "")
            data = __extract_json(raw)
            return data["p"] if data else -1
    except Exception:
        return -1


async def __classify_batch(rows):
    async with aiohttp.ClientSession() as session:
        tasks = [__classify_single(session, title, desc) for title, desc in rows]
        return await asyncio.gather(*tasks)


async def classify_by_local_model(df: pd.DataFrame):
    total_rows = len(df)
    logger.info(f"Classification started for {total_rows} rows")
    logger.debug(f"Using batch size: {config.LOCAL_MODEL_BATCH_SIZE}")

    rows = list(df[["Title", "Description"]].itertuples(index=False, name=None))

    results = []
    temp_file = os.path.join(config.EXPORT_FOLDER, "temp_results.csv")

    if os.path.exists(temp_file):
        logger.info(f"Found existing temp file at {temp_file}, loading…")
        temp_df = pd.read_csv(temp_file, sep=";")
        processed_indices = set(temp_df.index)
        results = temp_df["local_llm_points"].tolist()
        logger.info(f"Loaded {len(processed_indices)} previously processed rows")
    else:
        logger.info("No temp file found, starting fresh.")
        processed_indices = set()

    batch_size = config.LOCAL_MODEL_BATCH_SIZE
    num_batches = (total_rows + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, total_rows, batch_size), start=1):
        if i in processed_indices:
            logger.info(
                f"Batch {batch_idx}/{num_batches} starting at index {i} already processed"
            )
            continue

        end_index = min(i + batch_size, total_rows)

        batch = rows[i:end_index]
        batch_results = await __classify_batch(batch)

        results.extend(batch_results)

        temp_df = pd.DataFrame({"local_llm_points": results})
        temp_df.to_csv(temp_file, sep=";", index=False)
        logger.info(f"Batch {batch_idx}/{num_batches} completed")

    logger.info(f"Classification finished for all {total_rows} rows")
    return results
