from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "insurance_claims.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
