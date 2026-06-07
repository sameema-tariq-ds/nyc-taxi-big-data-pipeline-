"""
config.py — Single source of truth for all project constants, paths, and settings.

Design principles:
  - All paths are relative to this file's location (works on any machine)
  - Secrets come from .env (never hardcoded here)
  - Directories are auto-created on import
  - Every constant is typed and explained
"""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: locate project root and load .env
# ---------------------------------------------------------------------------

# ROOT_DIR is always the folder that contains this config.py file.
# This means the project works regardless of where it is cloned on any machine.
ROOT_DIR: Path = Path(__file__).resolve().parent

# Load .env file if it exists (won't error if missing — safe for CI/CD)
load_dotenv(ROOT_DIR / ".env")


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    """All filesystem paths used by the pipeline."""

    # Input
    raw_dir: Path = ROOT_DIR / "data" / "raw"  # downloaded .parquet files go here

    # Outputs
    logs_dir: Path = ROOT_DIR / "logs"  # pipeline run logs

    def __post_init__(self) -> None:
        """Auto-create every directory when config is imported.

        Senior pattern: never let a missing folder crash a pipeline mid-run.
        Fail fast on startup instead.
        """
        for path_field in self.__dataclass_fields__:
            path: Path = getattr(self, path_field)
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset Config
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """ "Configuration for selecting and naming NYC taxi parquet datasets."""

    # Which taxi type to use: "yellow", "green", or "fhv"
    taxi_type: str = "green"

    # Year to include: 2009 - 2026
    year_start: int = 2021
    year_end: int = 2025

    # Months to include. None = all 12.
    months: list[int] = field(default_factory=lambda: list(range(1, 13)))

    def parquet_filename(self, year: int, month: int) -> str:
        return f"{self.taxi_type}_tripdata_{year}-{month:02d}.parquet"


# ---------------------------------------------------------------------------
# Pipeline performance settings
# ---------------------------------------------------------------------------
@dataclass
class PipelineSettings:
    """Tuning knobs for memory and compute performance."""

    # Rows per chunk when reading with pandas.
    # At ~500 bytes/row for yellow taxi, 100k rows ≈ 50MB per chunk.
    # Increase if you have >16GB RAM, decrease if you hit MemoryError.
    chunk_size: int = 100_00


@dataclass
class SchemaSettings:
    """Column names, expected dtypes, and validation rules.

    Centralising this means if NYC TLC renames a column next year,
    you fix it in ONE place and the whole pipeline adapts.
    """

    # Target dtypes after memory optimisation.
    # The default int64/float64 pandas uses wastes 2–4x the memory needed.
    optimised_dtypes: dict[str, str] = field(
        default_factory=lambda: {
            "VendorID": "int8",  # max 2 ID
            "RatecodeID": "Int8",  # max 6
            "passenger_count": "Int8",  # max 6 passengers; int8 holds up to 127
            "PULocationID": "int16",  # 265 zones; int16 holds up to 32767
            "DOLocationID": "int16",
            "payment_type": "Int8",  # 1–6 categories
            "trip_distance": "float32",  # 4 decimal places is plenty
            "fare_amount": "float32",
            "tip_amount": "float32",
            "total_amount": "float32",
            "extra": "float32",
            "mta_tax": "float32",
            "tolls_amount": "float32",
            "improvement_surcharge": "float32",
            "congestion_surcharge": "float32",
            "cbd_congestion_fee": "float32",
            "trip_type": "Int8",
        }
    )

    # Categorical columns (convert to category dtype — saves ~90% memory on strings)
    categorical_columns: list[str] = field(
        default_factory=lambda: [
            "payment_type",
        ]
    )

    # Datetime columns to parse
    datetime_columns: list[str] = field(
        default_factory=lambda: [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ]
    )


# ---------------------------------------------------------------------------
# Top-level config object — this is what every module imports
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Master config. Import this everywhere: `from config import cfg`"""

    paths: Paths = field(default_factory=Paths)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    schema: SchemaSettings = field(default_factory=SchemaSettings)


# Module-level singleton — import this, not the class
cfg = Config()
