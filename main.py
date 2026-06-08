from logs_config import get_logger
from src.ingest import Ingestor
from src.optimize import MemoryOptimizer
from src.transform import Transformer

logger = get_logger(__name__)

if __name__ == "__main__":
    ingestor = Ingestor()
    available_raw_files = ingestor.list_available_files()
    logger.info(f"Available files: {(available_raw_files)}")

    df = ingestor.read_file(2025, 1)

    # list(ingestor.read_all())

    # list(ingestor.read_chunks(2025, 1))

    optimizer = MemoryOptimizer()
    df_opt, report = optimizer.optimize(df)
    logger.info(f"Memory Optimization Report:{report}")

    transformer = Transformer()
    df_transformed = transformer.transform(df_opt)

    logger.info(
        df_transformed[
            [
                "expected_fare",
                "fare_discrepancy",
                "avg_speed_mph",
                "is_invalid_ratecodeID",
            ]
        ].sample(3)
    )

    # 1. Check no nulls in your derived columns
    print(
        df_transformed[["trip_duration_min", "avg_speed_mph", "fare_discrepancy"]]
        .isnull()
        .sum()
    )

    # 2. Check no negative durations
    print(f"Negative durations: {(df_transformed['trip_duration_min'] < 0).sum()}")

    # 3. Check flag columns only contain 0 and 1
    flag_cols = [col for col in df_transformed.columns if col.startswith("is_")]
    for col in flag_cols:
        unique_vals = df_transformed[col].unique()
        print(f"{col}: {unique_vals}")
