
import argparse
import io
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except Exception:
    from transformers import AutoModelForVision2Seq as AutoVLMModel


MODEL_PATH = "/gemini/space/guarded_files/liujiang/privacy_aware_sft/outputs/sft/qwen3-vl-baseline/data_43526/merged/nonfreeze/merged_step_baseline_step2720"

DEFAULT_TASKS = {
    "falldown": {
        "model_path": MODEL_PATH,
        "test_path": "/gemini/space/guarded_files/liujiang/privacy_aware_sft/data/test.fall.parquet",
        "prompt": "请判断图片中是否有人跌倒。只回答：是 或 否。",
    },
    "smoke": {
        "model_path": MODEL_PATH,
        "test_path": "/gemini/space/guarded_files/liujiang/privacy_aware_sft/data/test.smoke.parquet",
        "prompt": "请判断图片中是否存在吸烟行为。只回答：是 或 否。",
    },
    "fight": {
        "model_path": MODEL_PATH,
        "test_path": "/gemini/space/guarded_files/liujiang/privacy_aware_sft/data/test.fight.fixed.parquet",
        "prompt": "请判断图片中是否存在打架行为。只回答：是 或 否。",
    },
}

IMAGE_COL_CANDIDATES = [
    "images",
    "image",
    "image_path",
    "img_path",
    "img",
    "path",
    "file",
    "filepath",
]
PROMPT_COL_CANDIDATES = ["prompt", "question", "instruction", "query", "text"]
LABEL_COL_CANDIDATES = ["label", "answer", "target", "gt", "ground_truth"]
MESSAGES_COL_CANDIDATES = ["messages", "conversations", "chat", "dialog"]

POS_WORDS = ["是", "有", "存在", "yes", "true", "1", "阳性"]
NEG_WORDS = ["否", "无", "没有", "none", "no", "false", "0", "阴性"]
TASK_CN_KEY = {"fight": "打架", "smoke": "抽烟", "falldown": "跌倒"}


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned Qwen3-VL models on parquet test sets"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["falldown", "smoke", "fight"],
        choices=list(DEFAULT_TASKS.keys()),
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit samples per task"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (1=single GPU, >1=multi-GPU with device_map)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--image_col", type=str, default="", help="Force image column name"
    )
    parser.add_argument(
        "--label_col", type=str, default="", help="Force label column name"
    )
    parser.add_argument(
        "--prompt_col", type=str, default="", help="Force prompt column name"
    )
    parser.add_argument(
        "--messages_col", type=str, default="", help="Force messages column name"
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="dataset",
        choices=["dataset", "custom"],
        help="dataset: use user prompt from messages/prompt col; custom: always use task default prompt",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="structured",
        choices=["structured", "binary"],
        help="structured: parse task label from '最终判断(...)'; binary: parse global yes/no from output",
    )
    parser.add_argument(
        "--model_path_override",
        type=str,
        default="",
        help="Override model path for all tasks",
    )
    parser.add_argument(
        "--save_details",
        type=str,
        default="",
        help="Optional path to save post-processed per-sample results jsonl",
    )
    parser.add_argument(
        "--raw_results_dir",
        type=str,
        default="",
        help="Directory to save raw inference outputs before metrics are computed",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Optional path to save summary metrics json",
    )
    parser.add_argument(
        "--badcase_dir",
        type=str,
        default="",
        help="Directory to save bad cases. Images are saved to badcase_dir/img/FP and badcase_dir/img/FN",
    )
    parser.add_argument(
        "--test_smoke_path", type=str, default="", help="Full path to smoke test set"
    )
    parser.add_argument(
        "--test_fall_path", type=str, default="", help="Full path to falldown test set"
    )
    parser.add_argument(
        "--test_fight_path", type=str, default="", help="Full path to fight test set"
    )
    parser.add_argument(
        "--external_lib",
        type=str,
        default="",
        help="External library to import for model patch (e.g., vision_tower_dp_everywhere)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--batch_size_falldown",
        type=int,
        default=0,
        help="Batch size for falldown task (0 means use --batch_size)",
    )
    parser.add_argument(
        "--batch_size_smoke",
        type=int,
        default=0,
        help="Batch size for smoke task (0 means use --batch_size)",
    )
    parser.add_argument(
        "--batch_size_fight",
        type=int,
        default=0,
        help="Batch size for fight task (0 means use --batch_size)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Global random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def has_model_weights(model_dir: Path) -> bool:
    weight_patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "*.safetensors",
        "*.bin",
        "model-*.safetensors",
        "pytorch_model-*.bin",
    ]
    for pat in weight_patterns:
        if any(model_dir.glob(pat)):
            return True
    if any(model_dir.glob("*.safetensors.index.json")):
        return True
    if any(model_dir.glob("*.bin.index.json")):
        return True
    return False


def find_latest_model_dir(model_root: str) -> str:
    root = Path(model_root)
    if (root / "config.json").exists() and has_model_weights(root):
        return str(root)

    candidates: List[Path] = []
    for cfg in root.rglob("config.json"):
        d = cfg.parent
        if has_model_weights(d):
            candidates.append(d)

    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)

    cfg_dirs = [p.parent for p in root.rglob("config.json")]
    if cfg_dirs:
        latest_cfg = max(cfg_dirs, key=lambda p: p.stat().st_mtime)
        raise FileNotFoundError(
            f"Found config but no model weights under: {model_root}. Latest config dir: {latest_cfg}"
        )

    raise FileNotFoundError(f"No config.json found under: {model_root}")


def first_existing_column(
    columns: Iterable[str], preferred: List[str]
) -> Optional[str]:
    columns_list = list(columns)
    mapping = {c.lower(): c for c in columns_list}
    for col in preferred:
        hit = mapping.get(col.lower())
        if hit is not None:
            return hit
    return None


def to_bool_label(value) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, (int, float)) and not pd.isna(value):
        return 1 if int(value) > 0 else 0

    text = str(value).strip().lower()
    if text == "":
        return None

    if any(w in text for w in ["yes", "true", "1", "是", "有", "存在", "阳性"]):
        return 1
    if any(w in text for w in ["no", "false", "0", "否", "无", "没有", "阴性"]):
        return 0
    return None


def normalize_binary_prediction(text: str) -> Optional[int]:
    if text is None:
        return None
    x = re.sub(r"\s+", "", str(text).strip().lower())

    if any(w in x for w in POS_WORDS):
        return 1
    if any(w in x for w in NEG_WORDS):
        return 0
    return None


def parse_messages_cell(value) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def extract_assistant_text(messages: List[dict]) -> str:
    if not messages:
        return ""
    for m in messages:
        if str(m.get("role", "")).lower() == "assistant":
            return str(m.get("content", ""))
    return ""


def extract_task_label_from_text(text: str, task_name: str) -> Optional[int]:
    if not text:
        return None

    m = re.search(r"最终判断\((.*?)\)", text)
    if m:
        inside = m.group(1)
        for part in re.split(r"[,，]", inside):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            if k.strip() == TASK_CN_KEY[task_name]:
                return to_bool_label(v.strip())

    k = TASK_CN_KEY[task_name]
    near = re.search(rf"{k}[^。；;,，\n]*[=:：]?(是|否)", text)
    if near:
        return to_bool_label(near.group(1))

    return None


def extract_label_from_messages(messages: List[dict], task_name: str) -> Optional[int]:
    return extract_task_label_from_text(extract_assistant_text(messages), task_name)


def _decode_image_from_dict(item: dict) -> Image.Image:
    if "bytes" in item and item["bytes"] is not None:
        raw = item["bytes"]
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        if isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw)).convert("RGB")
    if "path" in item and item["path"]:
        return Image.open(item["path"]).convert("RGB")
    raise ValueError(f"Unsupported image dict keys: {list(item.keys())}")


def load_image(row: pd.Series, image_col: str) -> Image.Image:
    value = row[image_col]

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        return _decode_image_from_dict(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty image list")
        first = value[0]
        if isinstance(first, dict):
            return _decode_image_from_dict(first)
        if isinstance(first, (bytes, bytearray)):
            return Image.open(io.BytesIO(first)).convert("RGB")

    raise ValueError(f"Unsupported image format in column '{image_col}': {type(value)}")


def extract_user_prompt_from_messages(messages: List[dict]) -> Optional[str]:
    if not messages:
        return None
    for m in messages:
        if str(m.get("role", "")).lower() == "user":
            text = str(m.get("content", "")).strip()
            if text:
                return text
    return None


def build_prompt(
    row: pd.Series,
    prompt_col: Optional[str],
    default_prompt: str,
    prompt_mode: str,
    messages: Optional[List[dict]] = None,
) -> str:
    if prompt_mode == "custom":
        return default_prompt

    if prompt_col and pd.notna(row[prompt_col]) and str(row[prompt_col]).strip():
        return str(row[prompt_col])

    if messages:
        msg_prompt = extract_user_prompt_from_messages(messages)
        if msg_prompt:
            return msg_prompt

    return default_prompt


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def get_task_batch_size(args: argparse.Namespace, task_name: str) -> int:
    task_batch_size = getattr(args, f"batch_size_{task_name}", 0)
    if task_batch_size == 0:
        task_batch_size = args.batch_size
    return task_batch_size


def predict_batch(
    model,
    processor,
    images: List[Image.Image],
    prompts: List[str],
    device: str,
    temperature: float,
    max_new_tokens: int,
    use_multi_gpu: bool = False,
) -> List[Tuple[str, float]]:
    if not images:
        return []

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image, prompt in zip(images, prompts)
    ]

    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    if use_multi_gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    preds = processor.batch_decode(new_tokens, skip_special_tokens=True)

    total_time = t1 - t0
    avg_time = total_time / len(images) if images else 0.0

    return [(pred.strip(), avg_time) for pred in preds]


def decode_prediction(raw_pred: str, task_name: str, eval_mode: str) -> Optional[int]:
    if eval_mode == "binary":
        return normalize_binary_prediction(raw_pred)
    return extract_task_label_from_text(raw_pred, task_name)


def infer_task_raw(
    task_name: str,
    model_path: str,
    test_path: str,
    default_prompt: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict]:
    resolved_model_path = find_latest_model_dir(model_path)
    print(f"\n===== [{task_name}] =====")
    print(f"Model: {resolved_model_path}")
    print(f"Data : {test_path}")

    df = pd.read_parquet(test_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    image_col = args.image_col or first_existing_column(
        df.columns, IMAGE_COL_CANDIDATES
    )
    prompt_col = args.prompt_col or first_existing_column(
        df.columns, PROMPT_COL_CANDIDATES
    )
    label_col = args.label_col or first_existing_column(
        df.columns, LABEL_COL_CANDIDATES
    )
    messages_col = args.messages_col or first_existing_column(
        df.columns, MESSAGES_COL_CANDIDATES
    )

    if image_col is None:
        raise KeyError(f"No image column found. Existing columns: {list(df.columns)}")
    if label_col is None and messages_col is None:
        raise KeyError(
            f"No label/messages column found. Existing columns: {list(df.columns)}"
        )

    dataset_meta = {
        "task_name": task_name,
        "test_path": str(test_path),
        "model_path": resolved_model_path,
        "image_col": image_col,
        "prompt_col": prompt_col,
        "label_col": label_col,
        "messages_col": messages_col,
        "prompt_mode": args.prompt_mode,
        "eval_mode": args.eval_mode,
        "max_samples": args.max_samples,
    }

    print(
        f"Columns -> image: {image_col}, prompt: {prompt_col}, "
        f"label: {label_col}, messages: {messages_col}"
    )
    print(f"Prompt mode: {args.prompt_mode}; Eval mode: {args.eval_mode}")

    if args.external_lib:
        import importlib

        print(f"[eval] Loading external_lib: {args.external_lib}")
        external_lib = importlib.import_module(args.external_lib)
        print(f"[eval] External_lib loaded: {external_lib}")

    dtype = resolve_dtype(args.dtype)
    if args.num_gpus > 1:
        model = AutoVLMModel.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoVLMModel.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        resolved_model_path, trust_remote_code=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    total_infer_time = 0.0
    total_attempted = 0
    raw_rows: List[Dict] = []

    task_batch_size = get_task_batch_size(args, task_name)
    print(f"[{task_name}] Using batch_size={task_batch_size}")

    batch_images: List[Image.Image] = []
    batch_prompts: List[str] = []
    batch_data: List[Dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {task_name}"):
        msgs: List[dict] = []
        if messages_col is not None:
            msgs = parse_messages_cell(row[messages_col])

        gt = None
        if label_col is not None:
            gt = to_bool_label(row[label_col])
        if gt is None and messages_col is not None:
            gt = extract_label_from_messages(msgs, task_name)
        if gt is None:
            continue

        image = load_image(row, image_col)
        prompt = build_prompt(
            row=row,
            prompt_col=prompt_col,
            default_prompt=default_prompt,
            prompt_mode=args.prompt_mode,
            messages=msgs,
        )

        batch_images.append(image)
        batch_prompts.append(prompt)
        batch_data.append(
            {
                "idx": int(idx),
                "gt": int(gt),
                "prompt": prompt,
                "source_row": {
                    "index": int(idx),
                    "task": task_name,
                },
            }
        )

        if len(batch_images) == task_batch_size:
            results = predict_batch(
                model=model,
                processor=processor,
                images=batch_images,
                prompts=batch_prompts,
                device=args.device,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_multi_gpu=(args.num_gpus > 1),
            )

            for i, (raw_pred, elapsed) in enumerate(results):
                data = batch_data[i]
                total_infer_time += elapsed
                total_attempted += 1
                raw_rows.append(
                    {
                        "task": task_name,
                        "index": int(data["idx"]),
                        "label": int(data["gt"]),
                        "raw_pred": raw_pred,
                        "prompt": data["prompt"],
                        "eval_mode": args.eval_mode,
                        "elapsed_seconds": elapsed,
                    }
                )

            batch_images = []
            batch_prompts = []
            batch_data = []

    if batch_images:
        results = predict_batch(
            model=model,
            processor=processor,
            images=batch_images,
            prompts=batch_prompts,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_multi_gpu=(args.num_gpus > 1),
        )

        for i, (raw_pred, elapsed) in enumerate(results):
            data = batch_data[i]
            total_infer_time += elapsed
            total_attempted += 1
            raw_rows.append(
                {
                    "task": task_name,
                    "index": int(data["idx"]),
                    "label": int(data["gt"]),
                    "raw_pred": raw_pred,
                    "prompt": data["prompt"],
                    "eval_mode": args.eval_mode,
                    "elapsed_seconds": elapsed,
                }
            )

    avg_infer_time = total_infer_time / total_attempted if total_attempted else 0.0
    print(f"[{task_name}] raw_inference_finished, attempted={total_attempted}")
    print(f"[{task_name}] avg_infer_seconds={avg_infer_time:.4f}")
    return raw_rows, dataset_meta


def save_task_raw_results(
    task_name: str,
    raw_rows: List[Dict],
    dataset_meta: Dict,
    raw_results_dir: Optional[str],
) -> Optional[Path]:
    if not raw_results_dir:
        return None

    save_dir = Path(raw_results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_dir / f"{task_name}_raw_results.jsonl"

    rows_to_save = []
    for row in raw_rows:
        payload = dict(row)
        payload["_meta"] = dataset_meta
        rows_to_save.append(payload)

    write_jsonl(raw_path, rows_to_save)
    print(f"[{task_name}] Saved raw inference results to: {raw_path}")
    return raw_path


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_badcase_image(
    df: pd.DataFrame,
    image_col: str,
    row_index: int,
    save_path: Path,
) -> bool:
    try:
        row = df.loc[row_index]
        image = load_image(row, image_col)
        ensure_parent_dir(save_path)
        image.save(save_path)
        return True
    except Exception as e:
        print(f"[badcase] failed to save image for index={row_index}: {e}")
        return False


def export_badcases(
    task_name: str,
    raw_rows: List[Dict],
    dataset_meta: Dict,
    badcase_dir: str,
    args: argparse.Namespace,
) -> Dict[str, int]:
    if not badcase_dir:
        return {
            "fp_saved": 0,
            "fn_saved": 0,
            "parse_failed_saved": 0,
        }

    base_dir = Path(badcase_dir)
    fp_img_dir = base_dir / "img" / "FP"
    fn_img_dir = base_dir / "img" / "FN"
    fp_img_dir.mkdir(parents=True, exist_ok=True)
    fn_img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_meta["test_path"])
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    image_col = dataset_meta["image_col"]

    fp_manifest = []
    fn_manifest = []
    parse_failed_manifest = []
    fp_saved = 0
    fn_saved = 0

    for row in raw_rows:
        pred = decode_prediction(row["raw_pred"], task_name, row.get("eval_mode", args.eval_mode))
        gt = int(row["label"])
        row_index = int(row["index"])

        if pred is None:
            parse_failed_manifest.append(
                {
                    "task": task_name,
                    "index": row_index,
                    "label": gt,
                    "raw_pred": row["raw_pred"],
                    "prompt": row.get("prompt", ""),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                }
            )
            continue

        if gt == 0 and pred == 1:
            img_path = fp_img_dir / f"{task_name}_idx{row_index}_gt0_pred1.png"
            if save_badcase_image(df=df, image_col=image_col, row_index=row_index, save_path=img_path):
                fp_saved += 1
            fp_manifest.append(
                {
                    "task": task_name,
                    "index": row_index,
                    "label": gt,
                    "pred": pred,
                    "raw_pred": row["raw_pred"],
                    "prompt": row.get("prompt", ""),
                    "image_path": str(img_path),
                }
            )
        elif gt == 1 and pred == 0:
            img_path = fn_img_dir / f"{task_name}_idx{row_index}_gt1_pred0.png"
            if save_badcase_image(df=df, image_col=image_col, row_index=row_index, save_path=img_path):
                fn_saved += 1
            fn_manifest.append(
                {
                    "task": task_name,
                    "index": row_index,
                    "label": gt,
                    "pred": pred,
                    "raw_pred": row["raw_pred"],
                    "prompt": row.get("prompt", ""),
                    "image_path": str(img_path),
                }
            )

    write_jsonl(base_dir / f"{task_name}_FP.jsonl", fp_manifest)
    write_jsonl(base_dir / f"{task_name}_FN.jsonl", fn_manifest)
    write_jsonl(base_dir / f"{task_name}_parse_failed.jsonl", parse_failed_manifest)

    print(
        f"[{task_name}] badcases exported -> FP: {len(fp_manifest)} ({fp_saved} images), "
        f"FN: {len(fn_manifest)} ({fn_saved} images), "
        f"parse_failed: {len(parse_failed_manifest)}"
    )
    return {
        "fp_saved": fp_saved,
        "fn_saved": fn_saved,
        "parse_failed_saved": len(parse_failed_manifest),
    }


def summarize_raw_predictions(
    task_name: str,
    raw_rows: List[Dict],
    args: argparse.Namespace,
) -> Tuple[Metrics, List[Dict], Dict]:
    metrics = Metrics()
    details: List[Dict] = []
    total_attempted = 0
    total_infer_time = 0.0
    parse_failed = 0

    for row in raw_rows:
        total_attempted += 1
        elapsed = float(row.get("elapsed_seconds", 0.0))
        total_infer_time += elapsed
        gt = int(row["label"])
        raw_pred = row["raw_pred"]
        pred = decode_prediction(raw_pred, task_name, row.get("eval_mode", args.eval_mode))

        detail = {
            "task": task_name,
            "index": int(row["index"]),
            "label": gt,
            "raw_pred": raw_pred,
            "prompt": row.get("prompt", ""),
            "eval_mode": row.get("eval_mode", args.eval_mode),
            "elapsed_seconds": elapsed,
        }

        if pred is None:
            parse_failed += 1
            detail["pred"] = None
            detail["status"] = "parse_failed"
            details.append(detail)
            continue

        detail["pred"] = int(pred)
        detail["status"] = "ok"

        if gt == 1 and pred == 1:
            metrics.tp += 1
            detail["case_type"] = "TP"
        elif gt == 0 and pred == 1:
            metrics.fp += 1
            detail["case_type"] = "FP"
        elif gt == 0 and pred == 0:
            metrics.tn += 1
            detail["case_type"] = "TN"
        elif gt == 1 and pred == 0:
            metrics.fn += 1
            detail["case_type"] = "FN"

        details.append(detail)

    avg_infer_time = total_infer_time / total_attempted if total_attempted else 0.0
    parse_failed_rate = parse_failed / total_attempted if total_attempted else 0.0

    extra_stats = {
        "attempted": total_attempted,
        "evaluated": metrics.total,
        "parse_failed": parse_failed,
        "parse_failed_rate": parse_failed_rate,
        "avg_infer_seconds": avg_infer_time,
    }

    print(
        f"[{task_name}] attempted={total_attempted}, evaluated={metrics.total}, "
        f"parse_failed={parse_failed} ({parse_failed_rate:.2%})"
    )
    print(
        f"[{task_name}] acc={metrics.accuracy:.4f}, precision={metrics.precision:.4f}, "
        f"recall={metrics.recall:.4f}, f1={metrics.f1:.4f}"
    )
    print(f"[{task_name}] avg_infer_seconds={avg_infer_time:.4f}")

    return metrics, details, extra_stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"=== Global seed: {args.seed} ===")

    task_path_map = {
        "smoke": args.test_smoke_path,
        "falldown": args.test_fall_path,
        "fight": args.test_fight_path,
    }

    print("=== 测试集信息 ===")
    for task in args.tasks:
        test_path = task_path_map.get(task, "")
        if not test_path:
            test_path = DEFAULT_TASKS[task]["test_path"]
        filename = Path(test_path).name
        df = pd.read_parquet(test_path)
        total_samples = len(df)
        if args.max_samples is not None:
            total_samples = min(total_samples, args.max_samples)
        print(f"[{task}] test_path={filename}, total_samples={total_samples}")

    all_results = {}
    all_details: List[Dict] = []

    for task in args.tasks:
        cfg = DEFAULT_TASKS[task]
        if task_path_map.get(task):
            cfg = dict(cfg)
            cfg["test_path"] = task_path_map[task]

        model_path = args.model_path_override or cfg["model_path"]

        raw_rows, dataset_meta = infer_task_raw(
            task_name=task,
            model_path=model_path,
            test_path=cfg["test_path"],
            default_prompt=cfg["prompt"],
            args=args,
        )

        raw_path = save_task_raw_results(
            task_name=task,
            raw_rows=raw_rows,
            dataset_meta=dataset_meta,
            raw_results_dir=args.raw_results_dir,
        )

        rows_for_metrics = raw_rows
        if raw_path is not None:
            rows_for_metrics = load_jsonl(raw_path)

        metrics, details, extra_stats = summarize_raw_predictions(
            task_name=task,
            raw_rows=rows_for_metrics,
            args=args,
        )

        badcase_stats = export_badcases(
            task_name=task,
            raw_rows=rows_for_metrics,
            dataset_meta=dataset_meta,
            badcase_dir=args.badcase_dir,
            args=args,
        )

        all_results[task] = {
            "attempted": extra_stats["attempted"],
            "evaluated": extra_stats["evaluated"],
            "parse_failed": extra_stats["parse_failed"],
            "parse_failed_rate": extra_stats["parse_failed_rate"],
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
            "avg_infer_seconds": extra_stats["avg_infer_seconds"],
            "eval_mode": args.eval_mode,
            "prompt_mode": args.prompt_mode,
            "raw_results_path": str(raw_path) if raw_path is not None else "",
            "badcase_dir": args.badcase_dir,
            "badcase_stats": badcase_stats,
        }
        all_details.extend(details)

    print("\n===== Summary =====")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        ensure_parent_dir(metrics_path)
        metrics_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved metrics json to: {metrics_path}")

    if args.save_details:
        save_path = Path(args.save_details)
        write_jsonl(save_path, all_details)
        print(f"Saved post-processed details to: {save_path}")


if __name__ == "__main__":
    main()
