import os


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODEL_DIR, "data")


def data_path(*parts: str) -> str:
    return os.path.join(DATA_DIR, *parts)


def run_path(run_name: str, *parts: str) -> str:
    return os.path.join(DATA_DIR, run_name, *parts)
