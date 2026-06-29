"""Decoupled persistence: generation appends one JSON record per (item x condition) as it
completes; analysis (#8) only ever reads these files -- it never touches a model.

Layout per run:
  <path>.jsonl           one JSON record per line (the full per-item trace)
  <path>.jsonl.meta.json sidecar header: full config + git hash + seed
"""
import json
import os
import subprocess


def meta_path(path: str) -> str:
    return path + ".meta.json"


def git_hash() -> str:
    """Current commit hash, or 'unknown' if not in a git repo / git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_meta(path: str, meta: dict) -> None:
    """Write the run header sidecar (config + git hash + seed). Overwrites on each (re)start."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def read_meta(path: str):
    p = meta_path(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_records(path: str) -> list:
    """All records in the file (skips blank lines). Empty list if the file does not exist."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def seen_item_ids(path: str) -> set:
    """item_ids already persisted for this output file -- used to skip them on resume."""
    return {r["item_id"] for r in read_records(path) if "item_id" in r}


class JsonlWriter:
    """Append-and-flush writer. Each record is flushed + fsync'd immediately so killing the
    process mid-run never loses an already-completed record."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
