"""
eda.py — Generate exploratory data analysis (EDA) reports and visualizations.

Responsibilities:
  1. Perform univariate analysis on numerical features using histograms and boxplots
  2. Analyze temporal trends such as trip counts by hour and average fares by weekday
  3. Explore relationships between key variables through bivariate scatter plots
  4. Save generated EDA reports as image files for downstream review

Usage:
    from eda import EDA
    eda = EDA()
    eda.display_eda_analysis(df)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import cfg
from logs_config import get_logger

logger = get_logger(__name__)


class EDA:
    """Generate and save exploratory data analysis visualizations."""

    def __init__(self) -> None:
        """Initialize output paths for EDA report images."""
        self.univariate_report_path = cfg.paths.reports_dir / "univariate_analysis.png"
        self.temporal_report_path = cfg.paths.reports_dir / "temporal_analysis.png"
        self.bivariate_report_path = cfg.paths.reports_dir / "bivariate_analysis.png"

    def display_eda_analysis(self, df: pd.DataFrame) -> None:
        """Run all EDA analyses and save their corresponding reports."""
        self._univariate_analysis(df)
        self._temporal_analysis(df)
        self._bivariate_analysis(df)

    def _univariate_analysis(self, df: pd.DataFrame) -> None:
        """Generate distribution and outlier visualizations for numeric features."""
        n_cols = 2
        n_rows = len(cfg.eda_config.numeric_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

        for i, col in enumerate(cfg.eda_config.numeric_cols):
            if col not in df.columns:
                continue

            sns.histplot(df[col], bins=50, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"Distribution of {col}")

            sns.boxplot(x=df[col], ax=axes[i, 1])
            axes[i, 1].set_title(f"Boxplot of {col}")

            plt.subplots_adjust(top=0.85)

        plt.tight_layout()
        fig.savefig(self.univariate_report_path, dpi=300, bbox_inches="tight")
        logger.info(f"Univariate analysis saved to {self.univariate_report_path.name}")
        plt.close(fig)

    def _temporal_analysis(self, df: pd.DataFrame) -> None:
        """Analyze trip patterns across pickup hours and weekdays."""

        if "pickup_hour" not in df.columns or "pickup_dayofweek" not in df.columns:
            logger.warning(
                "Temporal analysis requires 'pickup_hour' and 'pickup_dayofweek'"
                " columns. Skipping ..."
            )
            return

        if "trip_duration_min" not in df.columns or "fare_amount" not in df.columns:
            logger.warning(
                "Temporal analysis requires 'trip_duration_min' and 'fare_amount'"
                " columns. Skipping ..."
            )
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        df.groupby("pickup_hour")["trip_duration_min"].count().plot(
            kind="bar", ax=axes[0], title="Trip Count by Hour"
        )
        axes[0].set_xlabel("Pickup Hour")
        axes[0].set_ylabel("Trip Count")

        df.groupby("pickup_dayofweek")["fare_amount"].mean().rename(
            index=cfg.eda_config.days_names
        ).plot(kind="bar", ax=axes[1], title="Avg Fare by Day of Week")
        axes[1].set_xlabel("Day of Week")
        axes[1].set_ylabel("Average Fare ($)")

        plt.tight_layout()

        fig.savefig(self.temporal_report_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    def _bivariate_analysis(self, df: pd.DataFrame) -> None:
        """Analyze trip patterns across pickup hours and weekdays."""

        if "trip_distance" not in df.columns or "fare_amount" not in df.columns:
            logger.warning(
                "Bivariate analysis requires 'trip_distance' and 'fare_amount'"
                " columns. Skipping ..."
            )
            return

        if (
            "trip_duration_min" not in df.columns
            or "fare_discrepancy" not in df.columns
        ):
            logger.warning(
                "Bivariate analysis requires 'trip_duration_min' and 'fare_discrepancy'"
                " columns. Skipping ..."
            )
            return

        fig, axes = plt.subplots(3, 1, figsize=(5, 16))

        # 1. Distance vs Fare — should be linear
        axes[0].scatter(df["trip_distance"], df["fare_amount"], alpha=0.1, s=1)
        axes[0].set_title("Trip Distance vs Fare Amount")
        axes[0].set_xlabel("Trip Distance")
        axes[0].set_ylabel("Fare Amount")
        axes[0].grid(True)
        # 2. Duration vs Distance — should correlate
        axes[1].scatter(df["trip_duration_min"], df["trip_distance"], alpha=0.1, s=1)
        axes[1].set_title("Trip Duration vs Trip Distance")
        axes[1].set_xlabel("Trip Duration (min)")
        axes[1].set_ylabel("Trip Distance")
        axes[1].grid(True)
        # 3. Fare vs Discrepancy — non-zero discrepancy at high fares = fraud signal
        axes[2].scatter(df["fare_amount"], df["fare_discrepancy"], alpha=0.1, s=1)
        axes[2].set_title("Fare vs Discrepancy")
        axes[2].set_xlabel("Fare Amount")
        axes[2].set_ylabel("Fare Discrepancy")
        axes[2].grid(True)

        plt.tight_layout()
        fig.savefig(self.bivariate_report_path, dpi=300, bbox_inches="tight")
        logger.info(f"Bivariate analysis saved to {self.bivariate_report_path.name}")
        plt.close(fig)
