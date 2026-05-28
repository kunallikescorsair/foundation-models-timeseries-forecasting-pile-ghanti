"""
Project-level configuration for selected datasets and domain assignments.

This configuration uses compute-feasible datasets under 2MB for local
experiments, parameter tuning, and foundation-model fine-tuning attempts.

The original proposal included larger datasets, but several were not
practical to run locally within the available timeline.
"""

from __future__ import annotations


FINANCE_DATASETS = [
    "bitcoin",
    "fred_md",
    "m1_monthly",
]


ENERGY_DATASETS = [
    "solar_weekly",
    "saugeenday",
    "electricity_weekly",
]


RETAIL_DATASETS = [
    "car_parts",
    "nn5_weekly",
    "nn5_daily",
]


SELECTED_DATASETS = (
    FINANCE_DATASETS
    + ENERGY_DATASETS
    + RETAIL_DATASETS
)


DOMAIN_DATASETS = {
    "finance": FINANCE_DATASETS,
    "energy": ENERGY_DATASETS,
    "retail": RETAIL_DATASETS,
}


DATASET_TO_DOMAIN = {
    dataset_key: "finance" for dataset_key in FINANCE_DATASETS
} | {
    dataset_key: "energy" for dataset_key in ENERGY_DATASETS
} | {
    dataset_key: "retail" for dataset_key in RETAIL_DATASETS
}


def get_selected_datasets() -> list[str]:
    return list(SELECTED_DATASETS)


def get_domain_datasets(domain: str) -> list[str]:
    domain = domain.strip().lower()

    if domain not in DOMAIN_DATASETS:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available domains: {sorted(DOMAIN_DATASETS.keys())}"
        )

    return DOMAIN_DATASETS[domain].copy()


def get_dataset_domain(dataset_key: str) -> str:
    if dataset_key not in DATASET_TO_DOMAIN:
        raise ValueError(f"Dataset key is not part of project selection: {dataset_key}")

    return DATASET_TO_DOMAIN[dataset_key]


def validate_selected_dataset(dataset_key: str) -> None:
    if dataset_key not in SELECTED_DATASETS:
        raise ValueError(
            f"Dataset key is not part of the selected project datasets: {dataset_key}"
        )