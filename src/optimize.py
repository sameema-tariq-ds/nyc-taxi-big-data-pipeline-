"""
optimize.py — Memory optimization for pandas DataFrames.

The single most impactful thing for a big data pandas pipeline.
pandas defaults every integer to int64 (8 bytes) and every float to float64
(8 bytes). For this dataset, most columns fit in int8/int16/float32, cutting
memory by 50–70%.

Techniques used:
  1. Drop column that contains NaN values entirely.
  2. Downcast integers to the smallest type that fits the data range
  3. Downcast floats from float64 → float32
  4. Convert low-cardinality string/int columns to category dtype

Usage:
    from optimize import MemoryOptimizer
    optimizer = MemoryOptimizer()
    df_opt = optimizer.optimize(df)
    report  = optimizer.report(df_before, df_after)
"""

import numpy as np
import pandas as pd

from config import cfg
from logs_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Memory reporter — measures before/after and logs the result
# ---------------------------------------------------------------------------


def memory_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory usage in MB (including index, deep=True for objects)."""
    total_bytes: float = float(df.memory_usage(deep=True).sum())
    return total_bytes / 1024**2


def memory_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Build a human-readable memory comparison report.
    Returns a dict so callers can log it, save it to JSON, or assert on it
    in tests — don't just print inside the function (that would be untestable).
    """
    before_mb = memory_mb(df_before)
    after_mb = memory_mb(df_after)

    reduction = (1 - after_mb / before_mb) * 100 if before_mb > 0 else 0

    report = {
        "before_mb": round(before_mb, 2),
        "after_mb": round(after_mb, 2),
        "reduction_percent": round(reduction, 2),
        "rows": len(df_after),
        "columns": len(df_after.columns),
        "dtype_before": df_before.dtypes.astype(str).to_dict(),
        "dtype_after": df_after.dtypes.astype(str).to_dict(),
    }

    logger.info(
        f"Memory: {before_mb:.1f} MB -> {after_mb:.1f} MB"
        f"({reduction:.1f}%) reduction, {len(df_after):,} rows)"
    )

    return report


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------


class MemoryOptimizer:
    """Reduces DataFrame memory footprint without losing information.
    Design decision: we use explicit dtype targets from cfg rather than
    letting pandas auto-downcast. Auto-downcast can silently overflow if
    a future dataset has values outside the expected range. Explicit targets
    make the assumptions visible and testable.
    """

    def optimize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Run all optimizations and return (optimized_df, report_dict).
        Returns both the DataFrame AND the report so the caller can log or
        store the report without side effects inside this function.
        """
        df_before = df.copy()
        df_after = df.copy()

        df_after = self.drop_null_columns(df_after)
        df_after = self.parse_datetimes(df_after)
        df_after = self.downcast_int_floats(df_after)
        df_after = self.convert_categorical(df_after)

        report = memory_report(df_before, df_after)
        return df_after, report

    def drop_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that contain only null values.
        Fully-null columns add memory overhead and provide no analytical value.
        Some TLC datasets include optional fields that are entirely empty for
        certain years or taxi types. Removing them simplifies downstream
        processing and reduces DataFrame size.
        """
        try:
            null_cols = df.columns[df.isnull().all()].tolist()
            if not null_cols:
                logger.info("No fully-null columns found.")
                return df

            df = df.drop(columns=null_cols)  # no inplace — reassign instead
            logger.info(f"Dropped {len(null_cols)} fully-null column(s): {null_cols}")
            return df

        except Exception as e:
            logger.error(f"drop_null_columns failed: {e}", exc_info=True)
            raise

    def parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime columns if they arrived as strings or objects.
        Parquet usually preserves datetime types — but some older TLC files
        store timestamps as strings. This handles both cases safely.
        """
        for col in cfg.schema.datetime_columns:
            if col not in df.columns:
                continue
            if df[col].dtype == "object":
                logger.debug(f"Parsing {col} from string to datetime")
                df[col] = pd.to_datetime(df[col], errors="coerce")
            # Already datetime64 — nothing to do
        return df

    def verify_precision_loss(
        self,
        df: pd.DataFrame,
        col: str,
        target_dtype: str,
    ) -> pd.Series:
        """Verify that a dtype conversion does not alter data values.
        Downcasting can reduce memory usage significantly, but may introduce
        precision loss or overflow if the target dtype cannot accurately
        represent the original values. This method performs a round-trip
        conversion check and preserves the original data when a loss is detected.
        """
        try:
            before = df[col].copy()
            converted = before.astype(target_dtype)

            # pd.api.types understands both numpy and pandas extension types
            if pd.api.types.is_float_dtype(pd.api.types.pandas_dtype(target_dtype)):
                back = converted.astype(before.dtype)
                loss = not np.allclose(
                    before.dropna(),
                    back.dropna(),
                    rtol=1e-4,
                    atol=1e-6,
                    equal_nan=False,
                )
            else:
                # Integers (int8, Int8, Int16, etc.) — strict round-trip
                loss = not before.equals(converted.astype(before.dtype))

            if loss:
                logger.warning(
                    f"Precision loss detected in '{col}' casting to "
                    f"{target_dtype}. Keeping original ({before.dtype})."
                )
                return before

            logger.debug(f"'{col}': no precision loss casting to {target_dtype}.")
            return converted

        except Exception as e:
            logger.error(f"verify_precision_loss failed on '{col}': {e}", exc_info=True)
            raise

    def downcast_int_floats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast integer columns to the smallest safe type.
        We use explicit targets from cfg, not pd.to_numeric(downcast=...),
        because auto-downcast doesn't account for nullable integers (Int8 vs int8)
        and can silently lose data if future files exceed the expected range.
        Safe guard: we verify no values exceed the target type's bounds
        before casting. If they do, we log a warning and keep the original.
        Downcast float64 → float32 for numeric columns.
        float32 gives ~7 significant decimal digits — more than enough for
        dollar amounts and distances. We lose nothing meaningful.
        """
        skipped, converted, failed = [], [], []

        for col, target_dtype in cfg.schema.optimised_dtypes.items():
            if col not in df.columns:  # bug fix: was `in`, skipping everything
                logger.warning(f"'{col}' in schema but not in DataFrame — skipping.")
                skipped.append(col)
                continue

            try:
                df[col] = self.verify_precision_loss(df, col, target_dtype)
                converted.append(col)

            except Exception as e:
                logger.error(f"Failed to cast '{col}' to {target_dtype}: {e}")
                failed.append(col)  # don't raise — continue other columns

        logger.info(
            f"dtype conversion complete — "
            f"converted: {len(converted)}, "
            f"skipped: {len(skipped)}, "
            f"failed: {len(failed)}"
        )
        if failed:
            logger.warning(f"Columns that failed conversion: {failed}")

        return df

    def convert_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert low-cardinality columns to category dtype.
        category dtype stores one integer per row (the code) plus a small
        lookup table of unique values. For a column with 6 unique values
        across 3M rows, this cuts memory from ~24MB (int32) to ~3MB.
        Rule of thumb: use category when unique_values / total_rows < 0.5%
        """
        for col in cfg.schema.categorical_columns:
            if col not in df.columns:
                logger.warning(f"'{col}' in schema but not in DataFrame — skipping.")
                continue

            n_unique = df[col].nunique()
            ratio = n_unique / len(df) if len(df) > 0 else 0

            if ratio < 0.005:
                df[col] = df[col].astype("category")
                logger.debug(f"  {col}: → category ({n_unique} unique values)")
            else:
                logger.warning(
                    f"  {col}: skipping category conversion "
                    f"(cardinality {n_unique}/{len(df)} = {ratio:.3%} is too high)"
                )
        return df
