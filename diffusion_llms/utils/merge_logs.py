"""Utilities for merging rank-specific log files from multi-GPU runs."""

import json
from pathlib import Path
from typing import Union


def merge_rank_logs(
    log_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*_rank*.jsonl"
) -> int:
    """Merge rank-specific JSONL log files into a single sorted file.

    In multi-GPU runs, each rank writes to its own log file (e.g., log_rank0.jsonl,
    log_rank1.jsonl). This function merges them into a single file sorted by
    example_id for easy analysis.

    Args:
        log_dir: Directory containing rank-specific log files
        output_path: Path for the merged output file
        pattern: Glob pattern to match rank log files (default: *_rank*.jsonl)

    Returns:
        Number of entries merged

    Example:
        >>> merge_rank_logs("results/logs/", "results/samples.jsonl")
        1000  # Number of entries merged
    """
    log_dir = Path(log_dir)
    output_path = Path(output_path)

    # Collect all entries from rank files
    all_entries = []
    rank_files = list(log_dir.glob(pattern))

    if not rank_files:
        # Try looking for a single non-rank file (single GPU case)
        single_files = list(log_dir.glob("*.jsonl"))
        rank_files = [f for f in single_files if "_rank" not in f.name]

    for log_file in rank_files:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        all_entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line in {log_file}: {e}")

    # Sort by example_id
    all_entries.sort(key=lambda x: x.get("example_id", 0))

    # Write merged file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(all_entries)


def verify_merged_logs(
    log_path: Union[str, Path],
    expected_count: int = None
) -> dict:
    """Verify integrity of merged log file.

    Args:
        log_path: Path to merged JSONL file
        expected_count: Expected number of entries (optional)

    Returns:
        Dict with verification results:
        - count: Number of entries
        - duplicates: List of duplicate example_ids
        - gaps: List of missing example_ids (if sequential expected)
        - valid: Boolean indicating if file is valid
    """
    log_path = Path(log_path)

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    example_ids = [e.get("example_id", -1) for e in entries]
    unique_ids = set(example_ids)

    # Find duplicates
    duplicates = []
    seen = set()
    for eid in example_ids:
        if eid in seen:
            duplicates.append(eid)
        seen.add(eid)

    # Find gaps (assuming sequential IDs from 0)
    gaps = []
    if example_ids:
        max_id = max(example_ids)
        expected_ids = set(range(max_id + 1))
        gaps = sorted(expected_ids - unique_ids)

    result = {
        "count": len(entries),
        "unique_count": len(unique_ids),
        "duplicates": duplicates,
        "gaps": gaps,
        "valid": len(duplicates) == 0 and len(gaps) == 0,
    }

    if expected_count is not None:
        result["expected_count"] = expected_count
        result["valid"] = result["valid"] and len(entries) == expected_count

    return result
