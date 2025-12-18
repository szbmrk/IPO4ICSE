import sys
import pymysql
from config_loader import config
from core.log import get_logger

logger = get_logger("DB Check")


def check_mysql_connection() -> bool:
    try:
        conn = pymysql.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            connect_timeout=5,
        )
        conn.close()
        logger.info("MySQL connection successful")
        return True
    except pymysql.Error as e:
        logger.error(f"MySQL connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while connecting to MySQL: {e}")
        return False


def check_mysql_server_running() -> bool:
    try:
        conn = pymysql.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            connect_timeout=5,
        )
        conn.close()
        logger.info("MySQL server is running")
        return True
    except pymysql.OperationalError as e:
        if "Can't connect" in str(e) or "Connection refused" in str(e):
            logger.error(
                f"MySQL server is not running or not accessible at {config.DB_HOST}"
            )
            return False
        elif "Access denied" in str(e):
            # Server is running but credentials might be wrong
            logger.warning(
                "MySQL server is running but access denied - check credentials"
            )
            return True
        else:
            logger.error(f"MySQL connection error: {e}")
            return False
    except Exception as e:
        logger.error(f"Error checking MySQL server: {e}")
        return False


def ensure_mysql_connection() -> None:
    if not check_mysql_server_running():
        logger.error("=" * 60)
        logger.error("MySQL server is not running or not accessible!")
        logger.error("=" * 60)
        logger.error("Please ensure MySQL is running before continuing.")
        logger.error("")
        logger.error("To start MySQL:")
        logger.error("  - Linux: sudo systemctl start mysql")
        logger.error("  - macOS: brew services start mysql")
        logger.error("  - Or check if MySQL is running on the configured host")
        logger.error("")
        logger.error("Current configuration:")
        logger.error(f"  Host: {config.DB_HOST}")
        logger.error(f"  Database: {config.DB_NAME}")
        logger.error("=" * 60)
        sys.exit(1)

    if not check_mysql_connection():
        logger.error("=" * 60)
        logger.error("Cannot connect to MySQL database!")
        logger.error("=" * 60)
        logger.error("MySQL server is running but connection failed.")
        logger.error("")
        logger.error("Possible issues:")
        logger.error("  1. Database does not exist")
        logger.error("  2. Invalid credentials in .env file")
        logger.error("  3. User does not have permissions")
        logger.error("")
        logger.error("Current configuration:")
        logger.error(f"  Host: {config.DB_HOST}")
        logger.error(f"  User: {config.DB_USER}")
        logger.error(f"  Database: {config.DB_NAME}")
        logger.error("")
        logger.error("Please check your .env file and database configuration.")
        logger.error("=" * 60)
        sys.exit(1)
