
import os, csv, datetime
from typing import Dict, List

class CSVLogger:
    """
    Simple CSV logger that creates directory and header as needed, then appends rows.
    Example:
        logger = CSVLogger("logs/ner_wnut17_log.csv",
                           fieldnames=["epoch","split","loss","token_f1","entity_f1"])
        logger.log({"epoch":1,"split":"train","loss":1.23,"token_f1":0.78,"entity_f1":0.65})
    """
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp"] + self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict):
        row = dict(row)
        row["timestamp"] = datetime.datetime.utcnow().isoformat()
        # write only known fields
        out = {k: row.get(k, "") for k in ["timestamp"] + self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp"] + self.fieldnames)
            writer.writerow(out)
