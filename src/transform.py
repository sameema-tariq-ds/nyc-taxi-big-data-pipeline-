"""
transform.py — Feature engineering.

Takes the clean, memory-optimised DataFrame from optimize.py and adds
derived columns that power the downstream aggregation and anomaly modules.

All transformations are:
  - Vectorised (no Python loops over rows)
  - Null-safe (won't crash on NaN values)
  - Idempotent (safe to re-run on already-transformed data)

Features added:
  - trip_duration_sec     : pickup → dropoff in seconds


Usage:
    from transform import Transformer
    transformer = Transformer()
    df = transformer.transform(df)
"""

import pandas as pd

from config import cfg
from logs_config import get_logger

logger = get_logger(__name__)


class Transformer:

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps. Returns a new DataFrame."""
        df = df.copy()  # never mutate the input

        # Datetime Columns
        df = self._add_trip_duration(df)
        df = self._add_time_features(df)

        # Fare Columns
        df = self._calc_expected_fare(df)
        df = self._calc_fare_diff(df)

        # Distance and Location
        df = self._calc_avg_speed(df)
        df = self._validate_zone_id(df)

        # Passenger Count
        df = self._is_zero_passenger(df)

        # Rate COde ID
        df = self._is_valid_ratecodeid(df)

        # Payment Type
        df = self._calc_payment_anomalies(df)

        new_cols = [
            "trip_duration_min",
            "pickup_hour",
            "pickup_dayofweek",
            "pickup_month",
            "pickup_year",
            "expected_fare",
            "fare_discrepancy",
            "avg_speed_mph",
            "is_invalid_PULocation",
            "is_invalid_DOLocation",
            "is_same_zone",
            "is_zero_passenger",
            "is_invalid_ratecodeID",
            "is_invalid_payment",
            "is_cash_with_tip",
            "is_free_but_charged",
        ]

        added = [c for c in new_cols if c in df.columns]
        logger.info(f"Transform complete. Added {len(added)} features: {added}")
        return df

    # ------------------------------------------------------------------
    # Private feature methods
    # ------------------------------------------------------------------
    def _add_trip_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trip duration in seconds from pickup/dropoff timestamps.

        We use .dt.total_seconds() which handles DST and timezone edge cases
        correctly — unlike subtracting epoch integers manually.
        """
        pickup_col = "lpep_pickup_datetime"
        dropoff_col = "lpep_dropoff_datetime"

        if pickup_col not in df.columns or dropoff_col not in df.columns:
            logger.warning("Datetime columns missing; skipping duration feature.")
            return df

        df["trip_duration_min"] = (
            (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
        ).astype("float32")

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour, day-of-week, month, year from pickup timestamp.

        Using int8/int16 for these since they have tiny ranges.
        hour: 0–23 → int8
        dayofweek: 0–6 → int8
        month: 1–12 → int8
        year: 2009–2030 → int16
        """
        col = "lpep_pickup_datetime"
        if col not in df.columns:
            logger.warning("Pickup Datetime column missing; skipping time features.")
            return df

        pickup = df[col]
        df["pickup_hour"] = pickup.dt.hour.astype("int8")
        df["pickup_dayofweek"] = pickup.dt.dayofweek.astype("int8")
        df["pickup_month"] = pickup.dt.month.astype("int8")
        df["pickup_year"] = pickup.dt.year.astype("int16")

        return df

    def _calc_expected_fare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the expected trip fare from individual fare components.
        The TLC dataset stores (fare, taxes, tips, tolls, surcharges, etc.).
        Missing fare components are treated as zero because some columns
        are optional and may not exist in all dataset versions
        (e.g. cbd_congestion_fee).
        """
        cols = [c for c in cfg.transform_config.fare_columns if c in df.columns]

        df["expected_fare"] = df[cols].fillna(0).sum(axis=1).astype("float32")

        return df

    def _calc_fare_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the difference between reported and reconstructed fares.
        The expected_fare column is derived by summing all available fare
        components (base fare, taxes, tolls, tips, surcharges, etc.).
        Comparing it against the reported fare_amount helps identify
        inconsistencies, missing charges, rounding differences, or other
        potential data quality issues.
        """
        df["fare_discrepancy"] = (
            (df["fare_amount"] - df["expected_fare"]).abs().astype("float32")
        )

        return df

    def _calc_avg_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate average trip speed in miles per hour (mph).
        Average speed is derived from trip distance and trip duration,
        providing a useful feature for identifying traffic conditions,
        trip efficiency, and potential data quality issues.
        Formula:
            avg_speed_mph = trip_distance / (trip_duration_min / 60)
        Trips with zero or near-zero duration can produce infinite values.
        These are replaced with zero to avoid propagating invalid values
        into downstream analysis or machine learning pipelines.
        """
        distance_col = "trip_distance"
        duration_col = "trip_duration_min"

        if distance_col not in df.columns or duration_col not in df.columns:
            logger.warning("distance and time columns missing; skipping speed feature.")
            return df

        df["avg_speed_mph"] = (
            (df[distance_col] / (df[duration_col] / 60))
            .replace([float("inf"), -float("inf")], 0)
            .astype("float32")
        )

        df["avg_speed_mph"] = df["avg_speed_mph"].fillna(0).astype("float32")

        return df

    def _validate_zone_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate pickup and dropoff taxi zone identifiers.
        NYC TLC taxi trips should reference valid Taxi Zone IDs for both
        pickup and dropoff locations. Trips containing zone IDs outside
        the known TLC zone list are flagged as potentially invalid records.

        In addition, trips that start and end in the same zone are flagged.
        These are not necessarily erroneous, but may indicate very short
        trips, local travel patterns, or records worth further investigation.
        """
        if "PULocationID" not in df.columns or "DOLocationID" not in df.columns:
            logger.warning(
                "PULocationID and DOLocationID columns missing; "
                "skipping zone validation."
            )
            return df

        df["is_invalid_PULocation"] = (
            ~df["PULocationID"].isin(cfg.transform_config.valid_zone_ids)
        ).astype("int8")
        df["is_invalid_DOLocation"] = (
            ~df["DOLocationID"].isin(cfg.transform_config.valid_zone_ids)
        ).astype("int8")

        # Same pickup and dropoff zone — not always anomalous but worth flagging
        df["is_same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype("int8")

        return df

    def _is_zero_passenger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify trips with zero reported passengers.

        The TLC dataset occasionally contains trips with missing or zero
        passenger counts. A value of zero may represent data entry issues,
        shared rides with unreported counts, or other operational anomalies.
        Missing values are treated as zero to ensure consistent handling.
        """

        if "passenger_count" not in df.columns:
            logger.warning(
                "passenger_count column missing; skipping zero passenger feature."
            )
            return df

        df["passenger_count"] = df["passenger_count"].fillna(0).astype("Int8")
        df["is_zero_passenger"] = (df["passenger_count"] == 0).astype("int8")

        return df

    def _is_valid_ratecodeid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate TLC rate code identifiers.
        RatecodeID indicates the fare calculation method applied to a trip
        Trips with missing or unexpected rate codes are flagged for data
        quality review and downstream analysis.
        Missing RatecodeID values are filled with zero before validation so
        they can be consistently identified as invalid when compared against
        the list of known TLC rate codes.
        """

        if "RatecodeID" not in df.columns:
            logger.warning("RatecodeID column missing; skipping ratecode validation.")
            return df

        df["RatecodeID"] = df["RatecodeID"].fillna(0).astype("Int8")
        df["is_invalid_ratecodeID"] = (
            ~df["RatecodeID"].isin(cfg.transform_config.valid_rate_codes)
        ).astype("int8")

        return df

    def _calc_payment_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify payment-related anomalies and suspicious fare patterns.
        TLC trips include a payment type code indicating how the fare was
        settled (e.g., credit card, cash, no charge, disputed trip). This
        method validates payment types and flags combinations that may
        indicate data quality issues or unusual trip records.
        Checks performed:
            - Invalid payment type codes.
            - Cash trips with recorded tips.
            - No-charge or voided trips that still have a positive fare.
        """
        if (
            "payment_type" not in df.columns
            or "tip_amount" not in df.columns
            or "fare_amount" not in df.columns
        ):
            logger.warning(
                "payment_type, tip_amount, and fare_amount column missing; "
                "skipping payment validation."
            )
            return df

        df["is_invalid_payment"] = (
            ~df["payment_type"].isin(cfg.transform_config.valid_payment_types)
        ).astype("int8")

        # Cash payment with tip is suspicious — cash tips rarely get recorded
        df["is_cash_with_tip"] = (
            (df["payment_type"] == 2) & (df["tip_amount"] > 0)
        ).astype("int8")

        df["is_free_but_charged"] = (
            (df["payment_type"].isin([3, 6])) & (df["total_amount"] > 0)
        ).astype("int8")

        return df
