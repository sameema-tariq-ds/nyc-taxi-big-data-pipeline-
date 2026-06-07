from logs_config import get_logger
from src.ingest import Ingestor

logger = get_logger(__name__)

if __name__ == "__main__":
    ingestor = Ingestor()
    available_raw_files = ingestor.list_available_files()
    logger.info(f"Available files: {(available_raw_files)}")

    # ingestor.read_file(2025, 1)

    list(ingestor.read_all())

    list(ingestor.read_chunks(2025, 1))
