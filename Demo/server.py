"""
server.py — ANTIDOTE Live Demo API
Loads all 6 checkpoints (3 stages × 2 methods) on startup.
Exposes stage-aware /simulate endpoint so the frontend can walk
through the catastrophic-forgetting narrative interactively.
"""

import os
import sys
from typing import Literal

import torch
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import CFG, load_model, load_checkpoint, DEVICE, split_dataset  # type: ignore

# ── App & CORS ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ANTIDOTE Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Checkpoint paths ───────────────────────────────────────────────────────────
STAGES = ["after_t1", "after_t2", "after_t3"]

CHECKPOINT_PATHS = {
    "standard": {
        "after_t1": os.path.join(BASE_DIR, "results", "results-baselines", "checkpoints", "naive_sequential_after_T1_LLMail.pt"),
        "after_t2": os.path.join(BASE_DIR, "results", "results-baselines", "checkpoints", "naive_sequential_after_T2_HackAPrompt.pt"),
        "after_t3": os.path.join(BASE_DIR, "results", "results-baselines", "checkpoints", "naive_sequential_after_T3_BIPIA.pt"),
    },
    "antidote": {
        "after_t1": os.path.join(BASE_DIR, "results", "results-ewc+replay", "checkpoints", "ewc_plus_replay_after_T1_LLMail.pt"),
        "after_t2": os.path.join(BASE_DIR, "results", "results-ewc+replay", "checkpoints", "ewc_plus_replay_after_T2_HackAPrompt.pt"),
        "after_t3": os.path.join(BASE_DIR, "results", "results-ewc+replay", "checkpoints", "ewc_plus_replay_after_T3_BIPIA.pt"),
    },
}

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

# ── Load all 6 models ──────────────────────────────────────────────────────────
print("Loading checkpoints — this may take a moment...")
models: dict[str, dict[str, object]] = {"standard": {}, "antidote": {}}

for method, stages in CHECKPOINT_PATHS.items():
    for stage, ckpt_path in stages.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Check that your results folders are mounted correctly."
            )
        print(f"  Loading {method}/{stage} from {ckpt_path}")
        m = load_model()
        m = load_checkpoint(m, ckpt_path)
        m.eval()
        # Keep on CPU to avoid VRAM pressure from 6 simultaneous models
        m = m.cpu()
        models[method][stage] = m

print("All 6 checkpoints loaded.")

# Sanity check — make sure at least two models differ
w_std = models["standard"]["after_t3"].deberta.encoder.layer[0].attention.self.query_proj.weight[0, :3]
w_ant = models["antidote"]["after_t3"].deberta.encoder.layer[0].attention.self.query_proj.weight[0, :3]
if torch.allclose(w_std, w_ant):
    print("WARNING: standard/after_t3 and antidote/after_t3 have identical weights. Check paths.")
else:
    print("Checkpoint sanity check passed — models have different weights.")


# ── Test-set examples ──────────────────────────────────────────────────────────
EXAMPLES_DF: pd.DataFrame | None = None


def _load_examples() -> pd.DataFrame:
    data_dir = os.path.join(BASE_DIR, "pi-detection-data")
    files = [
        ("t1_llmail.parquet",      "t1_llmail"),
        ("t2_hackaprompt.parquet", "t2_hackaprompt"),
        ("t3_bipia.parquet",       "t3_bipia"),
    ]
    frames = []
    for fname, task in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found — skipping")
            continue
        df = pd.read_parquet(fpath)
        _, _, test_df = split_dataset(df, val_split=CFG["val_split"], test_split=CFG["test_split"])
        test_df = test_df[["text", "label"]].copy()
        test_df["task"] = task
        frames.append(test_df)
    if not frames:
        raise RuntimeError("No parquet files found in pi-detection-data/")
    return pd.concat(frames, ignore_index=True)


EXAMPLES_DF = _load_examples()
print(f"Loaded {len(EXAMPLES_DF)} test examples across 3 tasks.")


# ── Inference ──────────────────────────────────────────────────────────────────
def classify(model, text: str, threshold: float = 0.5) -> dict:
    """Run binary classification. Label 1 = injection (BLOCKED)."""
    model_on_device = model.to(DEVICE)
    enc = tokenizer(
        text,
        max_length=CFG["max_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model_on_device(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

    model.cpu()  # return to CPU immediately to free VRAM
    torch.cuda.empty_cache()

    prob_injection = probs[1]
    pred_label     = 1 if prob_injection >= threshold else 0

    return {
        "status":          "BLOCKED" if pred_label == 1 else "BYPASSED",
        "probs":           probs,
        "pred_label":      pred_label,
        "prob_injection":  round(prob_injection, 4),
    }


# ── Request / response models ──────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    text:  str
    stage: Literal["after_t1", "after_t2", "after_t3"] = "after_t3"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/simulate")
async def simulate(req: SimulateRequest):
    """
    Classify text with both models at the requested training stage.
    Returns BLOCKED/BYPASSED + probabilities for standard and antidote.
    """
    std_result = classify(models["standard"][req.stage], req.text)
    ant_result = classify(models["antidote"][req.stage], req.text)
    return {
        "stage":    req.stage,
        "standard": std_result,
        "antidote": ant_result,
    }


@app.post("/simulate_all_stages")
async def simulate_all_stages(req: SimulateRequest):
    """
    Run the same text through all 6 checkpoints at once.
    Used to build the forgetting-curve chart on the frontend.
    """
    results = {}
    for stage in STAGES:
        std = classify(models["standard"][stage], req.text)
        ant = classify(models["antidote"][stage], req.text)
        results[stage] = {"standard": std, "antidote": ant}
    return {"text": req.text, "results": results}


@app.get("/examples")
async def get_example(
    kind:                 str  = Query("attack", description="'attack' or 'any'"),
    task:                 str  = Query("any",   description="'t1_llmail', 't2_hackaprompt', 't3_bipia', or 'any'"),
    require_standard_fail: bool = Query(False,  description="Only return examples where naive model fails"),
    stage:                str  = Query("after_t3", description="Which stage checkpoint to test standard model against"),
):
    """Return a random test-set example, optionally filtered."""
    if EXAMPLES_DF is None or EXAMPLES_DF.empty:
        return {"error": "No examples loaded"}

    df = EXAMPLES_DF.copy()

    if kind == "attack":
        df = df[df["label"] == 1]
    if task != "any":
        df = df[df["task"] == task]
    if df.empty:
        return {"error": "No examples matching filters"}

    if not require_standard_fail:
        row = df.sample(1).iloc[0]
        return {"text": row["text"], "label": int(row["label"]), "task": row["task"]}

    for _ in range(30):
        row      = df.sample(1).iloc[0]
        gt_label = int(row["label"])
        res      = classify(models["standard"][stage], row["text"])
        if res["pred_label"] != gt_label:
            return {
                "text":                 row["text"],
                "label":                gt_label,
                "task":                 row["task"],
                "standard_pred_label":  res["pred_label"],
            }

    row = df.sample(1).iloc[0]
    return {"text": row["text"], "label": int(row["label"]), "task": row["task"]}


@app.get("/stats")
async def get_stats():
    """
    Return per-task F1 scores for each method at each stage.
    Pre-computed from your results JSON files so the frontend
    can show the forgetting curve without running inference on
    the whole test set live.
    """
    # Load from your saved results JSONs
    import json
    results = {}
    result_files = {
        "standard": os.path.join(BASE_DIR, "results", "results-baselines", "results", "results_baselines.json"),
        "antidote": os.path.join(BASE_DIR, "results", "results-ewc+replay", "results", "results_ewc_plus_replay.json"),
    }
    for method, path in result_files.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Extract results_matrix: results_matrix[i][j] = F1 on task j after training on task i
            key = "naive_sequential" if method == "standard" else "ewc_plus_replay"
            if key in data:
                results[method] = data[key].get("results_matrix", [])
        else:
            results[method] = []
    return results


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "models_loaded":   {m: list(stages.keys()) for m, stages in models.items()},
        "examples_loaded": len(EXAMPLES_DF) if EXAMPLES_DF is not None else 0,
    }