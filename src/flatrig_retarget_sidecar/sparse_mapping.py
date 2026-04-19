"""Heuristic sparse mapping generation for Motion2Motion-style retargeting."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flatrig_retarget_sidecar.spine_import import SpinePackage, load_spine_package

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback is covered indirectly
    linear_sum_assignment = None

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_CENTER = "center"
SIDE_UNKNOWN = "unknown"
TOKEN_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])|[^A-Za-z0-9]+")
LEFT_TOKENS = {"l", "lf", "left", "lhs"}
RIGHT_TOKENS = {"r", "rt", "right", "rhs"}
STOP_TOKENS = {
    "bone",
    "joint",
    "ctrl",
    "control",
    "deform",
    "def",
    "rig",
    "slot",
    "mesh",
    "attachment",
    "node",
}


@dataclass(slots=True)
class SparseChain:
    names: list[str]
    start_name: str
    end_name: str
    parent_name: str | None
    side: str
    is_main: bool
    attachment_index: int
    start_depth: int
    total_length: float
    span_length: float
    straightness: float
    centroid_x: float
    centroid_y: float
    direction_x: float
    direction_y: float
    slot_count: int
    name_tokens: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SparsePair:
    source: str
    target: str
    score: float
    reason: str


@dataclass(slots=True)
class SparseMappingSuggestion:
    source_name: str
    target_name: str
    root_joint: str
    mapping: list[dict[str, str]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def suggest_sparse_mapping(
    source: SpinePackage,
    target: SpinePackage,
    *,
    max_pairs: int = 12,
    min_chain_score: float = 0.38,
) -> SparseMappingSuggestion:
    source_root = _select_root(source)
    target_root = _select_root(target)
    source_chains = _collect_sparse_chains(source, source_root.name)
    target_chains = _collect_sparse_chains(target, target_root.name)

    main_source = next((chain for chain in source_chains if chain.is_main), None)
    main_target = next((chain for chain in target_chains if chain.is_main), None)

    branch_source = [chain for chain in source_chains if not chain.is_main]
    branch_target = [chain for chain in target_chains if not chain.is_main]

    direct_pairs, direct_score = _match_chain_pairs(
        branch_source,
        branch_target,
        mirror=False,
        min_score=min_chain_score,
    )
    mirrored_pairs, mirrored_score = _match_chain_pairs(
        branch_source,
        branch_target,
        mirror=True,
        min_score=min_chain_score,
    )
    use_mirror = mirrored_score > direct_score
    matched_chain_pairs = mirrored_pairs if use_mirror else direct_pairs

    anchor_pairs: list[SparsePair] = [
        SparsePair(
            source=source_root.name,
            target=target_root.name,
            score=1.0,
            reason="root",
        )
    ]

    if (
        main_source
        and main_target
        and main_source.end_name != source_root.name
        and main_target.end_name != target_root.name
    ):
        anchor_pairs.append(
            SparsePair(
                source=main_source.end_name,
                target=main_target.end_name,
                score=_score_chain_pair(main_source, main_target, mirror=use_mirror),
                reason="main_chain_leaf",
            )
        )

    for source_chain, target_chain, score in matched_chain_pairs:
        anchor_pairs.append(
            SparsePair(
                source=source_chain.start_name,
                target=target_chain.start_name,
                score=score,
                reason="chain_start",
            )
        )
        if (
            source_chain.end_name != source_chain.start_name
            and target_chain.end_name != target_chain.start_name
        ):
            anchor_pairs.append(
                SparsePair(
                    source=source_chain.end_name,
                    target=target_chain.end_name,
                    score=max(0.0, score - 0.05),
                    reason="chain_end",
                )
            )

    deduped_pairs = _dedupe_pairs(anchor_pairs, max_pairs=max_pairs)
    mapping = [{"source": pair.source, "target": pair.target} for pair in deduped_pairs]

    diagnostics = {
        "source_root": source_root.name,
        "target_root": target_root.name,
        "mirror_x": use_mirror,
        "source_chain_count": len(source_chains),
        "target_chain_count": len(target_chains),
        "chain_pairs": [
            {
                "source_chain": source_chain.names,
                "target_chain": target_chain.names,
                "score": round(score, 4),
            }
            for source_chain, target_chain, score in matched_chain_pairs
        ],
        "selected_pairs": [asdict(pair) for pair in deduped_pairs],
        "source_chains": [asdict(chain) for chain in source_chains],
        "target_chains": [asdict(chain) for chain in target_chains],
    }

    return SparseMappingSuggestion(
        source_name=_label_stem(source.source_label),
        target_name=_label_stem(target.source_label),
        root_joint=target_root.name,
        mapping=mapping,
        diagnostics=diagnostics,
    )


def build_motion2motion_mapping_payload(
    source: SpinePackage,
    target: SpinePackage,
    *,
    max_pairs: int = 12,
    min_chain_score: float = 0.38,
) -> dict[str, Any]:
    suggestion = suggest_sparse_mapping(
        source,
        target,
        max_pairs=max_pairs,
        min_chain_score=min_chain_score,
    )
    return {
        "source_name": suggestion.source_name,
        "target_name": suggestion.target_name,
        "root_joint": suggestion.root_joint,
        "mapping": suggestion.mapping,
    }


def _collect_sparse_chains(package: SpinePackage, root_name: str) -> list[SparseChain]:
    children = {bone.name: list(bone.children) for bone in package.bones}
    by_name = package.bones_by_name
    root = by_name[root_name]
    main_path = _compute_main_path(root_name, by_name, children)
    main_path_set = set(main_path)
    chains: list[SparseChain] = []

    if len(main_path) > 1:
        chains.append(
            _build_chain(
                package,
                names=main_path,
                root_name=root_name,
                is_main=True,
                attachment_index=0,
            )
        )

    for index, bone_name in enumerate(main_path):
        next_main = main_path[index + 1] if index + 1 < len(main_path) else None
        for child_name in children.get(bone_name, []):
            if child_name == next_main:
                continue
            chain_names = _walk_linear_chain(
                child_name,
                by_name,
                children,
                stop_names=main_path_set,
            )
            chains.append(
                _build_chain(
                    package,
                    names=chain_names,
                    root_name=root_name,
                    is_main=False,
                    attachment_index=index,
                )
            )

    if not chains:
        chains.append(
            _build_chain(
                package,
                names=[root.name],
                root_name=root_name,
                is_main=True,
                attachment_index=0,
            )
        )

    return chains


def _compute_main_path(
    root_name: str,
    by_name: dict[str, Any],
    children: dict[str, list[str]],
) -> list[str]:
    path = [root_name]
    current_name = root_name
    while children.get(current_name):
        ranked_children = sorted(
            children[current_name],
            key=lambda name: (
                int(by_name[name].descendant_count),
                int(by_name[name].child_count),
                int(by_name[name].slot_count),
                float(by_name[name].length),
            ),
            reverse=True,
        )
        best_child = ranked_children[0]
        path.append(best_child)
        current_name = best_child
    return path


def _walk_linear_chain(
    start_name: str,
    by_name: dict[str, Any],
    children: dict[str, list[str]],
    *,
    stop_names: set[str],
) -> list[str]:
    chain = [start_name]
    current_name = start_name
    while True:
        current_children = children.get(current_name, [])
        if len(current_children) != 1:
            break
        next_name = current_children[0]
        if next_name in stop_names:
            break
        chain.append(next_name)
        current_name = next_name
    return chain


def _build_chain(
    package: SpinePackage,
    *,
    names: list[str],
    root_name: str,
    is_main: bool,
    attachment_index: int,
) -> SparseChain:
    by_name = package.bones_by_name
    root = by_name[root_name]
    start = by_name[names[0]]
    end = by_name[names[-1]]
    points = [(by_name[name].world_x, by_name[name].world_y) for name in names]
    centroid_x = sum(point[0] for point in points) / max(len(points), 1)
    centroid_y = sum(point[1] for point in points) / max(len(points), 1)

    total_length = 0.0
    slot_count = 0
    for name in names:
        bone = by_name[name]
        total_length += max(
            float(bone.length), _distance((bone.world_x, bone.world_y), (bone.end_x, bone.end_y))
        )
        slot_count += int(bone.slot_count)

    start_point = (start.world_x, start.world_y)
    end_point = (end.end_x, end.end_y)
    span_length = _distance(start_point, end_point)
    straightness = span_length / max(total_length, 1e-6)
    direction_x, direction_y = _normalize_vector(
        end_point[0] - start_point[0],
        end_point[1] - start_point[1],
    )
    side = _infer_chain_side(
        names,
        by_name=by_name,
        root_x=float(root.world_x),
        bounds=package.setup_bounds,
        centroid_x=centroid_x,
    )

    name_tokens = []
    for name in names[:2] + names[-2:]:
        name_tokens.extend(_name_tokens(name))
    name_tokens = sorted(set(name_tokens))

    return SparseChain(
        names=list(names),
        start_name=start.name,
        end_name=end.name,
        parent_name=start.parent,
        side=side,
        is_main=is_main,
        attachment_index=int(attachment_index),
        start_depth=int(start.depth),
        total_length=float(total_length),
        span_length=float(span_length),
        straightness=float(straightness),
        centroid_x=float(centroid_x),
        centroid_y=float(centroid_y),
        direction_x=float(direction_x),
        direction_y=float(direction_y),
        slot_count=int(slot_count),
        name_tokens=name_tokens,
    )


def _match_chain_pairs(
    source_chains: list[SparseChain],
    target_chains: list[SparseChain],
    *,
    mirror: bool,
    min_score: float,
) -> tuple[list[tuple[SparseChain, SparseChain, float]], float]:
    if not source_chains or not target_chains:
        return [], 0.0

    if linear_sum_assignment is None:
        return _greedy_match_chain_pairs(
            source_chains, target_chains, mirror=mirror, min_score=min_score
        )

    cost_matrix = []
    score_matrix: list[list[float]] = []
    for source_chain in source_chains:
        cost_row = []
        score_row = []
        for target_chain in target_chains:
            score = _score_chain_pair(source_chain, target_chain, mirror=mirror)
            score_row.append(score)
            cost_row.append(1.0 - score)
        cost_matrix.append(cost_row)
        score_matrix.append(score_row)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs: list[tuple[SparseChain, SparseChain, float]] = []
    total_score = 0.0
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        score = score_matrix[row][col]
        if score < min_score:
            continue
        pairs.append((source_chains[row], target_chains[col], score))
        total_score += score
    return pairs, total_score


def _greedy_match_chain_pairs(
    source_chains: list[SparseChain],
    target_chains: list[SparseChain],
    *,
    mirror: bool,
    min_score: float,
) -> tuple[list[tuple[SparseChain, SparseChain, float]], float]:
    used_targets: set[int] = set()
    pairs: list[tuple[SparseChain, SparseChain, float]] = []
    total_score = 0.0
    for source_chain in source_chains:
        best_index = -1
        best_score = -1.0
        for index, target_chain in enumerate(target_chains):
            if index in used_targets:
                continue
            score = _score_chain_pair(source_chain, target_chain, mirror=mirror)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index < 0 or best_score < min_score:
            continue
        used_targets.add(best_index)
        pairs.append((source_chain, target_chains[best_index], best_score))
        total_score += best_score
    return pairs, total_score


def _score_chain_pair(source: SparseChain, target: SparseChain, *, mirror: bool) -> float:
    if source.is_main != target.is_main:
        return 0.0

    name_score = _token_similarity(source.name_tokens, target.name_tokens)
    length_score = _ratio_score(source.total_length, target.total_length)
    depth_score = 1.0 / (1.0 + abs(source.start_depth - target.start_depth))
    attachment_score = 1.0 / (1.0 + abs(source.attachment_index - target.attachment_index))
    straightness_score = 1.0 - min(abs(source.straightness - target.straightness), 1.0)
    slot_score = _ratio_score(max(source.slot_count, 1), max(target.slot_count, 1))

    target_side = target.side
    if mirror:
        if target_side == SIDE_LEFT:
            target_side = SIDE_RIGHT
        elif target_side == SIDE_RIGHT:
            target_side = SIDE_LEFT
    side_score = _side_similarity(source.side, target_side)

    direction_score = 0.5 + 0.5 * abs(
        source.direction_x * target.direction_x + source.direction_y * target.direction_y
    )

    score = (
        0.18 * name_score
        + 0.22 * length_score
        + 0.16 * depth_score
        + 0.16 * attachment_score
        + 0.10 * straightness_score
        + 0.08 * slot_score
        + 0.10 * side_score
        + 0.10 * direction_score
    )
    return float(max(0.0, min(1.0, score)))


def _dedupe_pairs(pairs: list[SparsePair], *, max_pairs: int) -> list[SparsePair]:
    seen_source: set[str] = set()
    seen_target: set[str] = set()
    deduped: list[SparsePair] = []
    for pair in sorted(
        pairs, key=lambda item: (-item.score, item.reason, item.target, item.source)
    ):
        if pair.source in seen_source or pair.target in seen_target:
            continue
        deduped.append(pair)
        seen_source.add(pair.source)
        seen_target.add(pair.target)
        if len(deduped) >= max(1, max_pairs):
            break
    deduped.sort(
        key=lambda item: (0 if item.reason == "root" else 1, item.reason, item.target, item.source)
    )
    return deduped


def _select_root(package: SpinePackage):
    roots = [bone for bone in package.bones if bone.parent is None]
    if not roots:
        raise ValueError(f"No root bone found for {package.source_label}")
    return max(
        roots,
        key=lambda bone: (
            int(bone.descendant_count),
            int(bone.child_count),
            int(bone.slot_count),
            float(bone.length),
        ),
    )


def _infer_chain_side(
    names: list[str],
    *,
    by_name: dict[str, Any],
    root_x: float,
    bounds: dict[str, float],
    centroid_x: float,
) -> str:
    side_votes = [_infer_side_from_name(name) for name in names]
    side_votes = [side for side in side_votes if side != SIDE_UNKNOWN]
    if side_votes:
        left_votes = sum(1 for side in side_votes if side == SIDE_LEFT)
        right_votes = sum(1 for side in side_votes if side == SIDE_RIGHT)
        if left_votes > right_votes:
            return SIDE_LEFT
        if right_votes > left_votes:
            return SIDE_RIGHT

    width = max(float(bounds.get("width", 0.0)), 1.0)
    delta_x = centroid_x - root_x
    threshold = max(0.05 * width, 0.5)
    if delta_x <= -threshold:
        return SIDE_LEFT
    if delta_x >= threshold:
        return SIDE_RIGHT

    if any(by_name[name].depth <= 1 for name in names):
        return SIDE_CENTER
    return SIDE_UNKNOWN


def _infer_side_from_name(name: str) -> str:
    tokens = _split_tokens(name)
    if any(token in LEFT_TOKENS for token in tokens):
        return SIDE_LEFT
    if any(token in RIGHT_TOKENS for token in tokens):
        return SIDE_RIGHT
    return SIDE_UNKNOWN


def _name_tokens(name: str) -> list[str]:
    tokens = []
    for token in _split_tokens(name):
        if token in LEFT_TOKENS or token in RIGHT_TOKENS or token in STOP_TOKENS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _split_tokens(name: str) -> list[str]:
    base = str(name or "").split(":")[-1]
    parts = [part.strip().lower() for part in TOKEN_SPLIT_RE.split(base) if part.strip()]
    return parts


def _token_similarity(source_tokens: list[str], target_tokens: list[str]) -> float:
    source_set = set(source_tokens)
    target_set = set(target_tokens)
    if not source_set or not target_set:
        return 0.0
    overlap = len(source_set & target_set)
    union = len(source_set | target_set)
    return overlap / max(union, 1)


def _ratio_score(a: float, b: float) -> float:
    safe_a = max(float(a), 1e-6)
    safe_b = max(float(b), 1e-6)
    return math.exp(-abs(math.log(safe_a / safe_b)))


def _side_similarity(a: str, b: str) -> float:
    if a == SIDE_UNKNOWN or b == SIDE_UNKNOWN:
        return 0.6
    if a == SIDE_CENTER and b == SIDE_CENTER:
        return 1.0
    if a == SIDE_CENTER or b == SIDE_CENTER:
        return 0.45
    return 1.0 if a == b else 0.1


def _normalize_vector(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-6:
        return 0.0, 1.0
    return x / length, y / length


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _label_stem(label: str) -> str:
    text = str(label or "").replace("\\", "/")
    if "!/" in text:
        text = text.split("!/", 1)[-1]
    return Path(text).stem or "skeleton"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sparse Motion2Motion mapping from two Spine skeletons."
    )
    parser.add_argument("source", help="Source Spine JSON/ZIP/SKEL path.")
    parser.add_argument("target", help="Target Spine JSON/ZIP/SKEL path.")
    parser.add_argument(
        "--output", default="", help="Optional path for the Motion2Motion mapping JSON."
    )
    parser.add_argument(
        "--diagnostics",
        default="",
        help="Optional path for a diagnostics JSON dump.",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=12, help="Maximum number of sparse pairs to emit."
    )
    parser.add_argument(
        "--min-chain-score",
        type=float,
        default=0.38,
        help="Minimum chain-match score kept after assignment.",
    )
    args = parser.parse_args()

    source = load_spine_package(args.source)
    target = load_spine_package(args.target)
    suggestion = suggest_sparse_mapping(
        source,
        target,
        max_pairs=max(1, int(args.max_pairs)),
        min_chain_score=float(args.min_chain_score),
    )
    payload = {
        "source_name": suggestion.source_name,
        "target_name": suggestion.target_name,
        "root_joint": suggestion.root_joint,
        "mapping": suggestion.mapping,
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

    if args.diagnostics:
        diagnostics_path = Path(args.diagnostics).expanduser().resolve()
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(suggestion.diagnostics, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
