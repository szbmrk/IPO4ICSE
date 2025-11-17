import asyncio
import aiohttp
from abc import ABC, abstractmethod
from core.log import get_logger
import json
import re
import pandas as pd
import os
from config_loader import config


def extract_json(text: str) -> int:
    try:
        s = text.decode() if isinstance(text, (bytes, bytearray)) else text
    except Exception:
        return -1

    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return -1

    blob = m.group(0)

    blob = blob.replace("'", '"')

    blob = re.sub(r"([\{,\s])(\w+)\s*:", r'\1"\2":', blob)

    try:
        data = json.loads(blob)
    except Exception:
        return -1

    p = data.get("p") if isinstance(data, dict) else None
    if isinstance(p, int) and 0 <= p <= 100:
        return p

    if isinstance(p, str):
        try:
            v = int(p)
            if 0 <= v <= 100:
                return v
        except Exception:
            return -1

    return -1


class BaseClassifier(ABC):
    def __init__(self, model_name, batch_size, timing_data):
        self.model_name = model_name
        self.batch_size = batch_size
        self.timing_data = timing_data

    @abstractmethod
    async def _get_model_name(self) -> str:
        pass

    @abstractmethod
    async def _classify_single(self, session, title, desc) -> int | None:
        pass

    async def _classify_batch(self, rows):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._classify_single(session, title, desc) for title, desc in rows
            ]
            results = await asyncio.gather(*tasks)

            processed = []
            for r in results:
                if isinstance(r, int):
                    processed.append(r if 0 <= r <= 100 else -1)
                    continue

                processed.append(-1)

            return processed

    async def _classify_dataframe(self, df):
        logger = get_logger(self.model_name)
        model_name = await self._get_model_name()
        points_column = f"{model_name}_validity_point"

        total_rows = len(df)
        logger.info(f"Classification started for {total_rows} rows")
        logger.debug(f"Using batch size: {self.batch_size}")

        rows = list(df[["Title", "Description"]].itertuples(index=False, name=None))

        results = []

        num_batches = (total_rows + self.batch_size - 1) // self.batch_size

        for batch_idx, i in enumerate(range(0, total_rows, self.batch_size), start=1):
            end_index = min(i + self.batch_size, total_rows)

            batch = rows[i:end_index]
            batch_results = await self._classify_batch(batch)

            results.extend(batch_results)

            logger.info(f"Batch {batch_idx}/{num_batches} completed")

        logger.info(f"Classification finished for all {total_rows} rows")

        timing_df = pd.DataFrame(
            {"Row": range(len(self.timing_data)), "Timing (s)": self.timing_data}
        )
        name = await self._get_model_name()
        timing_file_path = os.path.join(
            f"{config.EXPORT_FOLDER}/timing", f"{name}_timing.csv"
        )
        timing_df.to_csv(timing_file_path, index=False)
        logger.info(f"Saved local model timing data to {timing_file_path}")
        return results, points_column

    async def classify(self, df):
        return await self._classify_dataframe(df)
