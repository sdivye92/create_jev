import sys
sys.path.insert(0, "/home/divye/Documents/Projects/any2jev/src")
from pathlib import Path
from any2jev.data import save_jsonl, synthetic_records

out = Path("/home/divye/Documents/Projects/create_jev/data/synthetic_raw")
n = 3000
recs = synthetic_records(n, seed=0)
n_val = n_test = n // 10
print("train", save_jsonl(out / "train.jsonl", recs[: n - n_val - n_test]))
print("val  ", save_jsonl(out / "val.jsonl", recs[n - n_val - n_test : n - n_test]))
print("test ", save_jsonl(out / "test.jsonl", recs[n - n_test :]))
