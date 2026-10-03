import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def load(name: str) -> list[dict]:
    with open(DATA / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def solver_problems():
    return load("solver_problems.jsonl")


def nodered_requests():
    return load("nodered_requests.jsonl")
