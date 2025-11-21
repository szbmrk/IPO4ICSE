import os
from pathlib import Path
from dotenv import load_dotenv
import tomllib
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, env_file: str = ".env", toml_file: str = "config.toml"):
        self.env_path = Path(env_file)
        if self.env_path.exists():
            load_dotenv(dotenv_path=self.env_path)
        else:
            logger.error(f"{env_file} not found")
            exit(-1)

        self.toml_path = Path(toml_file)
        if self.toml_path.exists():
            with open(self.toml_path, "rb") as f:
                self.toml_config = tomllib.load(f)
        else:
            logger.error(f"{toml_file} not found")
            exit(-1)

        self._load_env()
        self._load_toml()

    def _load_env(self):
        self.DB_HOST = os.getenv("DB_HOST")
        self.DB_USER = os.getenv("DB_USER")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD")
        self.DB_NAME = os.getenv("DB_NAME")

    def _load_toml(self):
        export_cfg = self.toml_config.get("export", {})
        self.EXPORT_LIMIT = export_cfg.get("limit", 1000)
        self.EXPORT_FOLDER = os.getenv(
            "EXPORT_FOLDER", export_cfg.get("folder", "exports")
        )
        self.EXPORT_ENABLED = (
            os.getenv("EXPORT_ENABLED", str(export_cfg.get("enabled", True))).lower()
            == "true"
        )

        local_model_cfg = self.toml_config.get("local_model", {})
        self.LOCAL_MODEL_ENABLED = local_model_cfg.get("enabled", True)
        self.LOCAL_MODEL_URL = local_model_cfg.get("url", "http://localhost:8080")
        self.LOCAL_MODEL_BATCH_SIZE = local_model_cfg.get("batch_size", 16)
        self.LOCAL_MODEL_N_PREDICT = local_model_cfg.get("n_predict", 20)
        self.LOCAL_MODEL_TEMP = float(
            os.getenv("LOCAL_MODEL_TEMP", local_model_cfg.get("temp", 0.4))
        )

        benchmark_cfg = self.toml_config.get("benchmark", {})
        self.BENCHMARK_FOLDER = os.getenv(
            "BENCHMARK_FOLDER", benchmark_cfg.get("folder", "exports")
        )
        self.BENCHMARK_OUTPUT = os.getenv(
            "BENCHMARK_OUTPUT", benchmark_cfg.get("output", "benchmark")
        )

        logging_cfg = self.toml_config.get("logging", {})
        self.LOG_LEVEL = logging_cfg.get("level", "INFO")
        self.LOG_FORMAT = logging_cfg.get(
            "format", "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        self.LOG_DATEFMT = logging_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
        self.LOG_FILE = logging_cfg.get("file", None)


config = Config()
