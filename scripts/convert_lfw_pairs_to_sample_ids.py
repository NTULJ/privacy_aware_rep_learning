from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


NAME_NUM_PATTERN = re.compile(r"^(?P<name>.+?)_(?P<num>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LFW pair file into sample_id_a/sample_id_b format."
    )
    parser.add_argument(
        "--pair-file",
        type=Path,
        required=True,
        help="Input pair file (csv/jsonl).",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        required=True,
        help="Feature index.csv containing sample_id/image_path/label/person_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output normalized pair csv.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def parse_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("+-").isdigit():
        return int(text)
    return None


def parse_binary_label(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "same", "match", "positive"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "different", "mismatch", "negative"}:
        return 0
    parsed = parse_int(text)
    if parsed is None:
        return default
    return 1 if parsed != 0 else 0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def image_num_from_stem(stem: str) -> int | None:
    match = NAME_NUM_PATTERN.match(stem)
    if not match:
        return None
    return int(match.group("num"))


def guess_person_name(sample_id: str, image_path: str, label: str, person_id: str) -> str | None:
    for value in (person_id, label):
        value = (value or "").strip()
        if value:
            return normalize_name(value)
    if image_path:
        parent_name = Path(image_path).parent.name
        if parent_name:
            return normalize_name(parent_name)
    sample_match = NAME_NUM_PATTERN.match(sample_id)
    if sample_match:
        left = sample_match.group("name")
        # extractor default for LFW creates "<name>_<name>_0001"
        double_prefix = f"{left}_{left}"
        if sample_id.startswith(double_prefix + "_"):
            return normalize_name(left)
        return normalize_name(left)
    return None


def guess_image_num(sample_id: str, image_path: str) -> int | None:
    if image_path:
        path_num = image_num_from_stem(Path(image_path).stem)
        if path_num is not None:
            return path_num
    return image_num_from_stem(sample_id)


def build_lookup(index_csv: Path) -> tuple[dict[tuple[str, int], str], set[str]]:
    rows = read_csv_rows(index_csv)
    lookup: dict[tuple[str, int], str] = {}
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        sample_ids.add(sample_id)
        person_name = guess_person_name(
            sample_id=sample_id,
            image_path=row.get("image_path", ""),
            label=row.get("label", ""),
            person_id=row.get("person_id", ""),
        )
        image_num = guess_image_num(sample_id=sample_id, image_path=row.get("image_path", ""))
        if person_name is None or image_num is None:
            continue
        key = (person_name, image_num)
        # keep first appearance to preserve deterministic behavior
        lookup.setdefault(key, sample_id)
    return lookup, sample_ids


def resolve_sample_id(
    sample_or_name: str,
    image_num: int | None,
    lookup: dict[tuple[str, int], str],
    sample_ids: set[str],
) -> str | None:
    text = sample_or_name.strip()
    if not text:
        return None
    if text in sample_ids:
        return text
    if image_num is None:
        return None
    return lookup.get((normalize_name(text), image_num))


def parse_jsonl_pairs(pair_file: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in pair_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def parse_csv_pairs(pair_file: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with pair_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        raw_rows = list(reader)
    if not raw_rows:
        return rows

    header = [cell.strip().lower() for cell in raw_rows[0]]
    is_normalized = "sample_id_a" in header and "sample_id_b" in header
    if is_normalized:
        col_index = {name: idx for idx, name in enumerate(header)}
        for raw in raw_rows[1:]:
            sample_a = raw[col_index["sample_id_a"]].strip() if col_index["sample_id_a"] < len(raw) else ""
            sample_b = raw[col_index["sample_id_b"]].strip() if col_index["sample_id_b"] < len(raw) else ""
            pair_label = raw[col_index["pair_label"]].strip() if "pair_label" in col_index and col_index["pair_label"] < len(raw) else "0"
            split = raw[col_index["split"]].strip() if "split" in col_index and col_index["split"] < len(raw) else "all"
            if not sample_a or not sample_b:
                continue
            rows.append(
                {
                    "sample_id_a": sample_a,
                    "sample_id_b": sample_b,
                    "pair_label": parse_binary_label(pair_label, default=0),
                    "split": split or "all",
                }
            )
        return rows

    for raw in raw_rows[1:]:
        tokens = [token.strip() for token in raw if token.strip()]
        if not tokens:
            continue
        # Optional metadata row from original LFW pairs.txt conversion (e.g. "10,300").
        if len(tokens) == 2 and parse_int(tokens[0]) is not None and parse_int(tokens[1]) is not None:
            continue
        if len(tokens) == 3:
            name, num1, num2 = tokens
            i1 = parse_int(num1)
            i2 = parse_int(num2)
            if i1 is None or i2 is None:
                continue
            rows.append(
                {
                    "name_a": name,
                    "num_a": i1,
                    "name_b": name,
                    "num_b": i2,
                    "pair_label": 1,
                    "split": "all",
                }
            )
            continue
        if len(tokens) == 4:
            name_a, num_a, name_b, num_b = tokens
            i1 = parse_int(num_a)
            i2 = parse_int(num_b)
            if i1 is None or i2 is None:
                continue
            rows.append(
                {
                    "name_a": name_a,
                    "num_a": i1,
                    "name_b": name_b,
                    "num_b": i2,
                    "pair_label": 0,
                    "split": "all",
                }
            )
            continue
    return rows


def parse_pairs(pair_file: Path) -> list[dict[str, object]]:
    if pair_file.suffix.lower() == ".jsonl":
        return parse_jsonl_pairs(pair_file)
    return parse_csv_pairs(pair_file)


def pair_record_to_ids(
    row: dict[str, object],
    lookup: dict[tuple[str, int], str],
    sample_ids: set[str],
) -> tuple[str, str, int, str] | None:
    if "sample_id_a" in row and "sample_id_b" in row:
        sample_a = str(row.get("sample_id_a", "")).strip()
        sample_b = str(row.get("sample_id_b", "")).strip()
        if sample_a not in sample_ids or sample_b not in sample_ids:
            return None
        pair_label = parse_binary_label(row.get("pair_label", row.get("label", 0)), default=0)
        split = str(row.get("split", "all")).strip() or "all"
        return sample_a, sample_b, pair_label, split

    # Raw LFW pair format
    name_a = str(row.get("name_a", row.get("name", row.get("person_a", row.get("person1", ""))))).strip()
    name_b = str(row.get("name_b", row.get("person_b", row.get("person2", "")))).strip()
    num_a = parse_int(row.get("num_a", row.get("imagenum1", row.get("idx_a", row.get("index1")))))
    num_b = parse_int(row.get("num_b", row.get("imagenum2", row.get("idx_b", row.get("index2")))))
    if not name_b:
        # matched pairs may only provide one person name
        name_b = name_a
    if num_a is None or num_b is None:
        return None

    sample_a = resolve_sample_id(name_a, num_a, lookup, sample_ids)
    sample_b = resolve_sample_id(name_b, num_b, lookup, sample_ids)
    if sample_a is None or sample_b is None:
        return None

    default_label = 1 if normalize_name(name_a) == normalize_name(name_b) else 0
    pair_label = parse_binary_label(row.get("pair_label", row.get("label")), default=default_label)
    split = str(row.get("split", "all")).strip() or "all"
    return sample_a, sample_b, pair_label, split


def main() -> None:
    args = parse_args()
    lookup, sample_ids = build_lookup(args.index_csv)
    pair_rows = parse_pairs(args.pair_file)

    converted_rows: list[dict[str, object]] = []
    skipped = 0
    for row in pair_rows:
        resolved = pair_record_to_ids(row, lookup, sample_ids)
        if resolved is None:
            skipped += 1
            continue
        sample_a, sample_b, pair_label, split = resolved
        converted_rows.append(
            {
                "sample_id_a": sample_a,
                "sample_id_b": sample_b,
                "pair_label": int(pair_label),
                "split": split,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id_a", "sample_id_b", "pair_label", "split"],
        )
        writer.writeheader()
        writer.writerows(converted_rows)

    print(
        f"converted {len(converted_rows)} pairs to {args.output} "
        f"(input={len(pair_rows)}, skipped={skipped})"
    )


if __name__ == "__main__":
    main()
