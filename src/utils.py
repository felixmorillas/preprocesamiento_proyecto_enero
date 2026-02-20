
import pandas as pd
import yaml


def load_yaml(file_path: str) -> dict:

    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset(file_path: str) -> pd.DataFrame:

    return pd.read_csv(file_path)