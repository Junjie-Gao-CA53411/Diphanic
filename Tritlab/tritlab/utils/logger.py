"""实验登记簿：每次运行把 config + 结果追加到 jsonl。

半年后你想查"当时可学习 Delta 收敛到多少"，靠它。
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    def __init__(self, path: str = "logs/experiments.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, experiment: str, config: Dict[str, Any],
            results: Dict[str, Any], notes: str = "") -> None:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": experiment,
            "config": config,
            "results": results,
            "notes": notes,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[logger] appended -> {self.path}")

    def load(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
