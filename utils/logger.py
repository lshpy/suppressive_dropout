import os, json
import pandas as pd

def save_json(result: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

def save_logs_csv(logs: list, dirpath: str):
    os.makedirs(dirpath, exist_ok=True)
    pd.DataFrame(logs).to_csv(os.path.join(dirpath, "epoch_log.csv"), index=False)
