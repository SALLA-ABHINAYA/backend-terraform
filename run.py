import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "execution_times.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    encoding="utf-8",
)

logging.info("Test log entry - Logging is working!")
