import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset


# 预训练稳定版（推荐给当前 ~100M 模型）
# 目标：尽量使用“连续自然文本”而不是指令问答模板数据。
PRETRAIN_STABLE_PROFILE = [
    {
        "name": "Morton-Li/ChineseWebText2.0-HighQuality",
        "config": None,
        "split": "train",
        "max_samples": 1_600_000,
        "fields": ["text"],
    },
    {
        "name": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "max_samples": 900_000,
        "fields": ["text"],
    },
    {
        "name": "tiiuae/falcon-refinedweb",
        "config": None,
        "split": "train",
        "max_samples": 700_000,
        "fields": ["content"],
    },
    {
        "name": "dirtycomputer/weibo_senti_100k",
        "config": None,
        "split": "train",
        "max_samples": 120_000,
        "fields": ["review"],
    },
]

# 社交/指令风格数据（更适合 SFT 或风格注入，默认不建议作为主预训练）
SOCIAL_SFT_LIKE_PROFILE = [
    {
        "name": "wangrui6/Zhihu-KOL",
        "config": None,
        "split": "train",
        "max_samples": 200_000,
        "fields": ["INSTRUCTION", "INPUT", "RESPONSE"],
    },
    {
        "name": "roycehe/tieba",
        "config": None,
        "split": "train",
        "max_samples": 80_000,
        "fields": ["text"],
    },
    {
        "name": "pangjin001/xiaohongshu2",
        "config": None,
        "split": "train",
        "max_samples": 80_000,
        "fields": ["instruction", "input", "output"],
    },
]

# full 模式：尽可能全量（可通过 full_cap_per_dataset 控制上限）
PRETRAIN_FULL_PROFILE = [
    {
        "name": "Morton-Li/ChineseWebText2.0-HighQuality",
        "config": None,
        "split": "train",
        "max_samples": None,
        "fields": ["text"],
    },
    {
        "name": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "max_samples": None,
        "fields": ["text"],
    },
    {
        "name": "tiiuae/falcon-refinedweb",
        "config": None,
        "split": "train",
        "max_samples": None,
        "fields": ["content"],
    },
    {
        "name": "dirtycomputer/weibo_senti_100k",
        "config": None,
        "split": "train",
        "max_samples": None,
        "fields": ["review"],
    },
]

PROFILES = {
    "pretrain_stable": PRETRAIN_STABLE_PROFILE,
    "pretrain_full": PRETRAIN_FULL_PROFILE,
    "social_sft_like": SOCIAL_SFT_LIKE_PROFILE,
}


def normalize_text(text: str) -> str:
    text = str(text).replace("\r", "\n")
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    return text.strip()


def iter_texts(row: Dict, fields: List[str]) -> Iterable[str]:
    chunks: List[str] = []
    for f in fields:
        if f in row and row[f] is not None:
            val = row[f]
            if isinstance(val, str):
                chunks.append(val)
            elif isinstance(val, list):
                for x in val:
                    if isinstance(x, str):
                        chunks.append(x)

    if not chunks:
        return []

    merged = "\n".join(chunks)
    return [merged]


def load_stream(name: str, config: Optional[str], split: str):
    return load_dataset(name, config, split=split, streaming=True)


def resolve_max_samples(spec: Dict, sample_scale: float, full_cap_per_dataset: int):
    base_max_samples = spec.get("max_samples")
    if base_max_samples is None:
        return full_cap_per_dataset if full_cap_per_dataset > 0 else None
    return max(1, int(base_max_samples * sample_scale))


def build_mix(
    profile: str,
    out_path: str,
    min_chars: int = 16,
    sample_scale: float = 1.0,
    full_cap_per_dataset: int = 0,
    stats_path: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    specs = PROFILES[profile]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {}
    total_emitted = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for spec in specs:
            name = spec["name"]
            config = spec["config"]
            split = spec["split"]
            fields = spec["fields"]
            max_samples = resolve_max_samples(spec, sample_scale, full_cap_per_dataset)

            row_seen = 0
            text_seen = 0
            emitted = 0
            dropped_short = 0
            dropped_empty = 0

            try:
                ds = load_stream(name, config, split)
                for row in ds:
                    row_seen += 1
                    for text in iter_texts(row, fields):
                        text_seen += 1
                        text = normalize_text(text)
                        if not text:
                            dropped_empty += 1
                            continue
                        if len(text) < min_chars:
                            dropped_short += 1
                            continue

                        w.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        emitted += 1
                        total_emitted += 1

                        if max_samples is not None and emitted >= max_samples:
                            break

                    if max_samples is not None and emitted >= max_samples:
                        break

            except Exception as e:
                print(f"[WARN] skip dataset={name}, reason={e}")

            stats[name] = {
                "row_seen": row_seen,
                "text_seen": text_seen,
                "emitted": emitted,
                "dropped_short": dropped_short,
                "dropped_empty": dropped_empty,
                "max_samples": -1 if max_samples is None else max_samples,
            }
            print(
                f"[INFO] dataset={name}, row_seen={row_seen}, emitted={emitted}, "
                f"dropped_short={dropped_short}, dropped_empty={dropped_empty}"
            )

    stats["__total__"] = {
        "emitted": total_emitted,
        "datasets": len(specs),
    }

    print(f"[INFO] output={out_path}")
    print(f"[INFO] total_emitted={total_emitted}")

    if stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[INFO] stats={stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build pretraining JSONL mix")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["pretrain_stable", "pretrain_full", "social_sft_like"],
        default="pretrain_stable",
        help="pretrain_stable=预训练推荐, pretrain_full=尽可能全量, social_sft_like=偏SFT风格",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="dataset/pretrain_mix_pretrain_stable.jsonl",
        help="输出jsonl路径",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="dataset/pretrain_mix_stats.json",
        help="输出统计json路径",
    )
    parser.add_argument("--min_chars", type=int, default=16, help="最小文本长度")
    parser.add_argument(
        "--sample_scale",
        type=float,
        default=1.0,
        help="采样比例，例如0.1表示每个配置上限只取10%%",
    )
    parser.add_argument(
        "--full_cap_per_dataset",
        type=int,
        default=0,
        help="仅对 pretrain_full 生效；>0 时每个数据集最多写入该条数，0 表示不设上限",
    )

    args = parser.parse_args()
    build_mix(
        profile=args.profile,
        out_path=args.out_path,
        min_chars=args.min_chars,
        sample_scale=args.sample_scale,
        full_cap_per_dataset=args.full_cap_per_dataset,
        stats_path=args.stats_path,
    )

    # Workaround: in some Python 3.12 + datasets builds, interpreter teardown
    # can trigger a native crash after successful processing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
