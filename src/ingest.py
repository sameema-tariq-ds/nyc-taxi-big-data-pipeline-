"""
ingest.py — Validate and read raw NYC TLC parquet files.

Responsibilities:
  1. Validate schema: correct columns exist, dtypes are sensible
  2. Read them in chunks so RAM usage stays bounded
  3. Log bad/missing files without crashing the whole pipeline

Usage:
    from ingest import Ingestor
    ingestor = Ingestor()
    df = ingestor.read_file(2023, 1)           # single month
    df = ingestor.read_all()                   # all configured years/months
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config import cfg
from logs_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------
class SchemaValidator:
    """Validates that a parquet file has the columns and structure we expect.

    Why validate before reading?
    Catching this early prevents silent downstream bugs.
    """

    def validate_file(self, path: Path) -> bool:
        """Check parquet metadata without reading the full file into memory.
        check if file path exists
        check if it is actualy a file
        pq.read_schema() reads only the footer bytes of the parquet file —
        it's essentially free, no matter how large the file is.
        """
        logger.info("Starting validation")
        try:
            if not path.exists():
                logger.warning(f"File does not exist: {path}")
                return False

            if not path.is_file():
                logger.warning(f"Path is not a file: {path}")
                return False

            schema = pq.read_schema(path)

            if not set(schema.names):
                logger.warning(f"File {path} has no columns. Skipping.")
                return False

            logger.info(f"Schema OK. {path.name}")
            return True

        except Exception as e:
            logger.error(f"Cannot read schema from {path.name} : {e}")
            return False

    def available_columns(self, path: Path) -> list[str]:
        """Return columns that file actually has."""
        schema = pq.read_schema(path)
        return list(schema.names)


class Ingestor:
    """Reads raw parquet files into pandas DataFrames.
    Two reading modes:
      - read_file(): single month, returns one DataFrame
      - read_chunks(): single month, yields chunk-by-chunk (memory-safe)
      - read_all(): all months, yields one DataFrame per file
    All modes only load the columns defined in cfg.schema.columns_to_keep.
    """

    def __init__(self) -> None:
        self.validator = SchemaValidator()

    def read_file(self, year: int, month: int) -> pd.DataFrame | None:
        """Read a single monthly parquet file into a DataFrame.
        Uses pyarrow engine which supports:
          - Predicate pushdown (skip unneeded row groups)
          - Column pruning (only read columns we need)
          - Efficient null handling
        """
        path = cfg.paths.raw_dir / cfg.dataset_config.parquet_filename(year, month)

        if not self.validator.validate_file(path):
            return None

        try:
            logger.info(
                f"Reading file: {path.name} ({path.stat().st_size / 1e6:.1f} MB)..."
            )
            columns = self.validator.available_columns(path)
            df = pd.read_parquet(path, engine="pyarrow", columns=columns)
            logger.info(
                f"Loaded {len(df):,} rows x {len(df.columns)} columns from {path.name}"
            )
            logger.info(df.sample(3))
            return df

        except Exception as e:
            logger.error(f"Failed to read {path.name}: {e}")
            return None

    def read_chunks(self, year: int, month: int) -> Iterator[pd.DataFrame]:
        """Read a single file in chunks. Use when RAM is limited.
        Each chunk is a complete, valid DataFrame — safe to process independently.
        The chunk_size is set in cfg.pipeline.chunk_size.
        """
        path = cfg.paths.raw_dir / cfg.dataset_config.parquet_filename(year, month)

        if not self.validator.validate_file(path):
            return None

        logger.info(
            f"Reading file: {path.name} ({path.stat().st_size / 1e6:.1f} MB)..."
        )
        columns = self.validator.available_columns(path)

        # PyArrow ParquetFile gives us row-group-level iteration —
        # more efficient than pandas chunked CSV reading because
        # it skips decoding entire row groups we haven't requested.
        pf = pq.ParquetFile(path)
        chunk_size = cfg.pipeline.chunk_size
        chunk_num = 0

        for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
            df = batch.to_pandas()
            chunk_num += 1
            logger.info(
                f"Loaded {path.name}-chunk{chunk_num}: "
                f"{len(df):,} rows x {len(df.columns)} columns"
            )
            yield df

    def read_all(self) -> Iterator[pd.DataFrame]:
        """Iterate over all configured year/month files.
        Yields one DataFrame per file. The caller decides whether to
        concatenate them or process them one by one.
        yield instead of returning a giant concatenated
        DataFrame — lets the caller control memory usage.
        """

        logger.info("Reading all files...")

        year_start = cfg.dataset_config.year_start
        year_end = cfg.dataset_config.year_end
        months = cfg.dataset_config.months

        for year in range(year_start, year_end + 1):
            for month in months:
                df = self.read_file(year, month)
                if df is not None:
                    yield df

    def list_available_files(self) -> list[str]:
        """Return all parquet files currently in the raw data directory."""
        return [file.name for file in cfg.paths.raw_dir.glob("*.parquet")]
