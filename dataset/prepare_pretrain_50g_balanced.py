import argparse
import json
import os
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import load_dataset


ZH_SOURCES: List[Tuple[str, Optional[str], str, str]] = [
    # (dataset_name, config, split, text_field)
    ("Morton-Li/ChineseWebText2.0-HighQuality", None, "train", "text"),
]

EN_SOURCES: List[Tuple[str, Optional[str], str, str]] = [
    ("Skylion007/openwebtext", None, "train", "text"),
    ("tiiuae/falcon-refinedweb", None, "train", "content"),
]


def load_stream(name: str, config: Optional[str], split: str):
    return load_dataset(name, config, split=split, streaming=True)


def normalize_text(text: str) -> str:
    text = str(text).replace("\r", "\n")
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    return text.strip()


def count_cjk_chars(text: str) -> int:
    count = 0
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF):
            count += 1
    return count


def count_ascii_letters(text: str) -> int:
    return sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)


def is_valid_zh(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    cjk = count_cjk_chars(text)
    # Keep lines that are clearly Chinese-heavy.
    return cjk >= max(12, int(0.2 * len(text)))


def is_valid_en(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    letters = count_ascii_letters(text)
    cjk = count_cjk_chars(text)
    if letters < max(20, int(0.25 * len(text))):
        return False
    if cjk > int(0.03 * len(text)):
        return False
    return True


def iter_source_texts(
    sources: List[Tuple[str, Optional[str], str, str]],
    min_chars: int,
    lang: str,
    stats: Dict,
) -> Iterator[str]:
    for name, cfg, split, field in sources:
        stats["sources"][name] = {
            "rows_seen": 0,
            "rows_emitted": 0,
            "dropped_short_or_invalid": 0,
        }
        try:
            ds = load_stream(name, cfg, split)
            for row in ds:
                stats["sources"][name]["rows_seen"] += 1
                val = row.get(field)
                if not isinstance(val, str):
                    stats["sources"][name]["dropped_short_or_invalid"] += 1
                    continue
                text = normalize_text(val)
                ok = is_valid_zh(text, min_chars) if lang == "zh" else is_valid_en(text, min_chars)
                if not ok:
                    stats["sources"][name]["dropped_short_or_invalid"] += 1
                    continue
                stats["sources"][name]["rows_emitted"] += 1
                yield text
        except Exception as e:
            stats["source_errors"].append({"dataset": name, "error": str(e)})


def make_jsonl_line(text: str) -> bytes:
    return (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")


def build_balanced_dataset(
    out_path: str,
    stats_path: str,
    target_gb: float,
    min_chars: int,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    target_total_bytes = int(target_gb * (1024 ** 3))
    target_zh_bytes = target_total_bytes // 2
    target_en_bytes = target_total_bytes - target_zh_bytes

    stats: Dict = {
        "target_gb": target_gb,
        "target_total_bytes": target_total_bytes,
        "target_zh_bytes": target_zh_bytes,
        "target_en_bytes": target_en_bytes,
        "written_total_bytes": 0,
        "written_zh_bytes": 0,
        "written_en_bytes": 0,
        "written_lines": 0,
        "sources": {},
        "source_errors": [],
    }

    zh_iter = iter_source_texts(ZH_SOURCES, min_chars, "zh", stats)
    en_iter = iter_source_texts(EN_SOURCES, min_chars, "en", stats)

    need_zh = True
    need_en = True

    with open(out_path, "wb") as f:
        while need_zh or need_en:
            # Prefer the language currently further below its target ratio.
            zh_gap = target_zh_bytes - stats["written_zh_bytes"]
            en_gap = target_en_bytes - stats["written_en_bytes"]
            choose_lang = "zh" if zh_gap >= en_gap else "en"

            if choose_lang == "zh" and need_zh:
                try:
                    line = make_jsonl_line(next(zh_iter))
                    if stats["written_zh_bytes"] + len(line) <= target_zh_bytes:
                        f.write(line)
                        stats["written_zh_bytes"] += len(line)
                        stats["written_total_bytes"] += len(line)
                        stats["written_lines"] += 1
                    else:
                        need_zh = False
                except StopIteration:
                    need_zh = False
            elif choose_lang == "en" and need_en:
                try:
                    line = make_jsonl_line(next(en_iter))
                    if stats["written_en_bytes"] + len(line) <= target_en_bytes:
                        f.write(line)
                        stats["written_en_bytes"] += len(line)
                        stats["written_total_bytes"] += len(line)
                        stats["written_lines"] += 1
                    else:
                        need_en = False
                except StopIteration:
                    need_en = False
            else:
                # Fallback when one side is already done.
                if need_zh:
                    try:
                        line = make_jsonl_line(next(zh_iter))
                        if stats["written_zh_bytes"] + len(line) <= target_zh_bytes:
                            f.write(line)
                            stats["written_zh_bytes"] += len(line)
                            stats["written_total_bytes"] += len(line)
                            stats["written_lines"] += 1
                        else:
                            need_zh = False
                    except StopIteration:
                        need_zh = False
                if need_en:
                    try:
                        line = make_jsonl_line(next(en_iter))
                        if stats["written_en_bytes"] + len(line) <= target_en_bytes:
                            f.write(line)
                            stats["written_en_bytes"] += len(line)
                            stats["written_total_bytes"] += len(line)
                            stats["written_lines"] += 1
                        else:
                            need_en = False
                    except StopIteration:
                        need_en = False

    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print(f"[INFO] output={out_path}")
    print(f"[INFO] stats={stats_path}")
    print(f"[INFO] written_total_bytes={stats['written_total_bytes']}")
    print(f"[INFO] written_zh_bytes={stats['written_zh_bytes']}")
    print(f"[INFO] written_en_bytes={stats['written_en_bytes']}")
    print(f"[INFO] written_lines={stats['written_lines']}")


def main():
    parser = argparse.ArgumentParser(description="Build a 50/50 Chinese-English pretraining JSONL by size")
    parser.add_argument(
        "--out_path",
        type=str,
        default="dataset/pretrain_mix_50g_balanced.jsonl",
        help="Output jsonl path",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="dataset/pretrain_mix_50g_balanced_stats.json",
        help="Output stats json path",
    )
    parser.add_argument(
        "--target_gb",
        type=float,
        default=50.0,
        help="Target output dataset size in GiB",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=32,
        help="Minimum text length to keep",
    )

    args = parser.parse_args()

    build_balanced_dataset(
        out_path=args.out_path,
        stats_path=args.stats_path,
        target_gb=args.target_gb,
        min_chars=args.min_chars,
    )

    # Workaround for some Python 3.12 + datasets teardown crashes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
