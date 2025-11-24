import pandas as pd
import csv
from pandas.errors import ParserError
from core.log import get_logger

logger = get_logger("FileReader")


def safe_read_csv(file_path: str, default_sep: str = ";") -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, sep=default_sep)
    except ParserError as e1:
        logger.warning(
            f"Initial parse with sep='{default_sep}' failed for {file_path}: {e1}. Trying to sniff delimiter."
        )

    detected_sep = None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(10000)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=";,|\t")
            detected_sep = dialect.delimiter
            logger.info(f"Detected delimiter '{detected_sep}' for {file_path}")
    except Exception:
        logger.warning(
            f"Could not detect delimiter for {file_path}, will try alternate parsing strategies."
        )

    if detected_sep and detected_sep != default_sep:
        try:
            return pd.read_csv(file_path, sep=detected_sep)
        except ParserError:
            logger.warning(
                f"Parsing with detected delimiter '{detected_sep}' failed for {file_path}."
            )

    try:
        return pd.read_csv(
            file_path,
            sep=default_sep,
            quoting=csv.QUOTE_NONE,
        )
    except Exception:
        logger.warning(
            f"Parsing with QUOTE_NONE failed for {file_path}, attempting csv.reader fallback."
        )

    try:
        sep_to_use = detected_sep or default_sep
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=sep_to_use)
            rows = list(reader)

        if not rows:
            raise ValueError(f"No rows parsed from {file_path}")

        header = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=header)
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV {file_path}: {e}")
        raise
