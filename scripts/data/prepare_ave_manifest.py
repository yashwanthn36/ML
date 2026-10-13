"""Create a segment-level AVE manifest from the archive metadata."""
from __future__ import annotations
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "AVE" / "ave"
ARCHIVE = ROOT / "data" / "raw" / "AVE" / "ave.zip"
SPLIT_FILES = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}

def parse(content: str, name: str) -> list[dict[str, str]]:
    lines = content.lstrip("\ufeff").splitlines()
    records = []
    for line_num, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = [x.strip() for x in line.split("&")]
        if len(fields) != 5:
            raise ValueError(f"{name}:{line_num}: expected 5 fields, found {len(fields)}")
        category, video_id, quality, start, end = fields
        records.append({"category": category, "video_id": video_id, "quality": quality,
                        "start_time": start, "end_time": end})
    return records

def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as z:
        members = {Path(x.filename).name: x for x in z.infolist()}
        needed = {"Annotations.txt", "ReadMe.txt", *SPLIT_FILES.values()}
        if missing := needed - members.keys():
            raise ValueError(f"Missing archive metadata: {sorted(missing)}")
        contents = {name: z.read(members[name]).decode("utf-8-sig") for name in needed}
        annotations = parse(contents["Annotations.txt"], "Annotations.txt")
        videos = {Path(x.filename).stem for x in z.infolist()
                  if not x.is_dir() and x.filename.lower().endswith(".mp4")}
        splits = {s: parse(contents[f], f) for s, f in SPLIT_FILES.items()}
    classes = sorted({r["category"] for r in annotations})
    label_ids = {name: i for i, name in enumerate(classes)}
    manifest = []
    for split, rows in splits.items():
        for i, r in enumerate(rows, 1):
            manifest.append({"sample_id": f"{split}_{i:04d}", "video_id": r["video_id"],
                "label": r["category"], "label_id": label_ids[r["category"]], "quality": r["quality"],
                "split": split, "time_start": float(r["start_time"]), "time_end": float(r["end_time"]),
                "video_path": f"data/raw/AVE/ave/video/{r['video_id']}.mp4",
                "audio_path": f"data/raw/AVE/ave/audio/{r['video_id']}.wav",
                "video_present_in_archive": r["video_id"] in videos})
    with (DATA_DIR / "ave_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    split_stats = {}
    ids_by_split = {}
    for split, rows in splits.items():
        counts = Counter(r["category"] for r in rows)
        ids = [r["video_id"] for r in rows]
        ids_by_split[split] = set(ids)
        split_stats[split] = {"segment_count": len(rows), "unique_video_count": len(set(ids)),
                              "repeated_video_rows": len(ids)-len(set(ids)), "class_counts": dict(sorted(counts.items()))}
    overlaps = {f"{a}_{b}": sorted(ids_by_split[a] & ids_by_split[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    summary = {"archive": ARCHIVE.name, "archive_size_bytes": ARCHIVE.stat().st_size,
        "video_file_count": len(videos), "annotation_segment_count": len(annotations),
        "official_split_segment_count": len(manifest), "annotation_only_row_count": len(annotations)-len(manifest),
        "annotation_video_count": len({r["video_id"] for r in annotations}), "class_count": len(classes),
        "classes": classes, "split_summary": split_stats,
        "video_id_overlap_between_splits": {k: {"count": len(v), "video_ids": v} for k,v in overlaps.items()},
        "split_video_ids_missing_from_archive": {s: sorted({r["video_id"] for r in rows}-videos) for s,rows in splits.items()},
        "readme": contents["ReadMe.txt"].strip(),
        "notes": ["Rows are event segments; multiple segments can refer to one source video.",
                  "Official splits have some source-video ID overlap.",
                  "Audio paths are populated after running scripts/data/extract_ave_audio.py."]}
    (DATA_DIR / "ave_dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'ave_manifest.csv'} ({len(manifest)} segments)")
    print(f"Wrote {DATA_DIR / 'ave_dataset_summary.json'} ({len(classes)} classes, {len(videos)} videos)")

if __name__ == "__main__":
    main()





