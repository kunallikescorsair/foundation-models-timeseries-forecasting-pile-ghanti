"""
Project-level configuration for selected datasets and domain assignments.

This module centralizes:
- the selected datasets for the capstone project
- domain-level dataset groupings
- helper functions for validating dataset keys
"""

from __future__ import annotations


SELECTED_DATASETS = [
    "bitcoin",
    "fred_md",
    "m4_quarterly",
    "australian_electricity_demand",
    "electricity_weekly",
    "solar_10_minutes",
    "nn5_daily",
    "dominick",
    "car_parts",
]


FINANCE_DATASETS = [
    "bitcoin",
    "fred_md",
    "m4_quarterly",
]


ENERGY_DATASETS = [
    "australian_electricity_demand",
    "electricity_weekly",
    "solar_10_minutes",
]


RETAIL_DATASETS = [
    "nn5_daily",
    "dominick",
    "car_parts",
]


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
    """
    Return the full list of selected project datasets.
    """
    return SELECTED_DATASETS.copy()


def get_domain_datasets(domain: str) -> list[str]:
    """
    Return the selected datasets for a given domain.

    Parameters
    ----------
    domain : str
        Domain name: 'finance', 'energy', or 'retail'.

    Returns
    -------
    list[str]
        Dataset keys for the domain.
    """
    domain = domain.strip().lower()

    if domain not in DOMAIN_DATASETS:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available domains: {sorted(DOMAIN_DATASETS.keys())}"
        )

    return DOMAIN_DATASETS[domain].copy()


def get_dataset_domain(dataset_key: str) -> str:
    """
    Return the domain for a selected dataset key.

    Parameters
    ----------
    dataset_key : str
        Selected dataset key.

    Returns
    -------
    str
        Domain label.
    """
    if dataset_key not in DATASET_TO_DOMAIN:
        raise ValueError(f"Dataset key is not part of project selection: {dataset_key}")

    return DATASET_TO_DOMAIN[dataset_key]


def validate_selected_dataset(dataset_key: str) -> None:
    """
    Validate that a dataset key belongs to the project selection.
    """
    if dataset_key not in SELECTED_DATASETS:
        raise ValueError(
            f"Dataset key is not part of the selected project datasets: {dataset_key}"
        )