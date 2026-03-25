#!/usr/bin/env python3
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


RUN_START_RE = re.compile(
    r"^(?P<ts>\S+\s+\S+)\s+run_start(?:\s+run_label=(?P<run_label>\S+))?"
    r"(?:\s+n=(?P<n>\d+))?"
    r"(?:\s+chunksize=(?P<chunksize>\d+))?"
    r"(?:\s+run_mode=(?P<run_mode>\S+))?"
    r"(?:\s+temp_encoding_mode=(?P<encoding>\S+))?"
    r"(?:\s+commit_every_chunks=(?P<commit>\d+))?"
    r"(?:\s+benchmark_max_level=(?P<benchmark>\S+))?"
    r"(?:\s+test_up_to=(?P<test_up_to>\S+))?"
    r"(?:\s+run35_compat=(?P<run35_compat>\S+))?"
    r"(?:\s+max_squares=(?P<max_squares>\d+))?"
    r"(?:\s+global_coverage_prune=(?P<global_prune>\S+))?"
    r"(?:\s+edge_neighborhood_prune=(?P<edge_prune>\S+))?"
    r"(?:\s+edge_neighborhood_lookahead=(?P<edge_neighborhood_lookahead>\S+))?"
    r"(?:\s+edge_neighborhood_lookahead_node_budget=(?P<edge_neighborhood_lookahead_node_budget>\S+))?"
    r"(?:\s+prioritized_edge_single_pass=(?P<prioritized_edge_single_pass>\S+))?"
    r"(?:\s+check_vertex_links=(?P<check_links>\S+))?"
    r"(?:\s+reuse_vertex_codes=(?P<reuse_vertex_codes>\S+))?"
    r"(?:\s+fast_hash_dedup=(?P<fast_hash_dedup>\S+))?"
    r"(?:\s+blob_encode_cache=(?P<blob_encode_cache>\S+))?"
    r"(?:\s+stage_good_inserts=(?P<stage_good_inserts>\S+))?"
    r"(?:\s+bulk_label_dedup=(?P<bulk_label_dedup>\S+))?"
    r"(?:\s+frontier_sharding=(?P<frontier_sharding>\S+))?"
    r"(?:\s+frontier_shards=(?P<frontier_shards>\S+))?"
    r"(?:\s+frontier_sharding_min_level=(?P<frontier_sharding_min_level>\S+))?"
    r"(?:\s+faceperm_by_cube=(?P<faceperm_by_cube>\S+))?"
    r"(?:\s+batch_merge_inserts=(?P<batch_merge>\S+))?"
)
RUN_END_RE = re.compile(r"^\S+\s+\S+\s+run_end")
CHUNK_RE = re.compile(
    r"^\S+\s+\S+\s+level=(?P<level>\d+)\s+chunk=\d+/\d+\s+"
    r"read_ms=(?P<read>\d+)\s+analyze_ms=(?P<analyze>\d+)\s+reduce_ms=(?P<reduce>\d+)\s+"
    r"proc_ms=(?P<proc>\d+)\s+generated=(?P<generated>\d+)\s+labeled_ins=(?P<labeled>\d+)\s+"
    r"good_ins=(?P<good>\d+)"
)
COMMIT_RE = re.compile(
    r"^\S+\s+\S+\s+level=(?P<level>\d+)\s+commit_ms=(?P<commit_ms>\d+)\s+chunks_in_tx=(?P<chunks_in_tx>\d+)"
)
PRUNE_RE = re.compile(
    r"^\S+\s+\S+\s+level=(?P<level>\d+)\s+prune_global_coverage=(?P<global>\d+)\s+prune_edge_neighborhood=(?P<edge>\d+)"
)


@dataclass
class LevelStats:
    chunks: int = 0
    read_ms: int = 0
    analyze_ms: int = 0
    reduce_ms: int = 0
    proc_ms: int = 0
    commit_ms: int = 0
    generated: int = 0
    labeled_ins: int = 0
    good_ins: int = 0
    prune_global_coverage: int = 0
    prune_edge_neighborhood: int = 0

    @property
    def total_ms(self) -> int:
        return self.proc_ms + self.commit_ms


@dataclass
class RunStats:
    index: int
    ts: str
    run_label: str = "-"
    n: str = "Unknown"
    chunksize: str = "Unknown"
    run_mode: str = "Unknown"
    encoding: str = "Unknown"
    commit_every_chunks: str = "Unknown"
    benchmark_max_level: str = "Unknown"
    test_up_to: str = "Unknown"
    run35_compat: str = "Unknown"
    max_squares: str = "Unknown"
    global_coverage_prune: str = "Unknown"
    edge_neighborhood_prune: str = "Unknown"
    edge_neighborhood_lookahead: str = "Unknown"
    edge_neighborhood_lookahead_node_budget: str = "Unknown"
    prioritized_edge_single_pass: str = "Unknown"
    check_vertex_links: str = "Unknown"
    reuse_vertex_codes: str = "Unknown"
    fast_hash_dedup: str = "Unknown"
    blob_encode_cache: str = "Unknown"
    stage_good_inserts: str = "Unknown"
    bulk_label_dedup: str = "Unknown"
    frontier_sharding: str = "Unknown"
    frontier_shards: str = "Unknown"
    frontier_sharding_min_level: str = "Unknown"
    faceperm_by_cube: str = "Unknown"
    batch_merge_inserts: str = "Unknown"
    ended: bool = False
    levels: dict[int, LevelStats] = field(default_factory=dict)

    def level(self, lvl: int) -> LevelStats:
        if lvl not in self.levels:
            self.levels[lvl] = LevelStats()
        return self.levels[lvl]


def parse_log(path: Path) -> list[RunStats]:
    runs: list[RunStats] = []
    current: RunStats | None = None
    run_idx = 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RUN_START_RE.match(raw)
        if m:
            run_idx += 1
            current = RunStats(
                index=run_idx,
                ts=m.group("ts"),
                run_label=(m.group("run_label") or "-"),
                n=(m.group("n") or "Unknown"),
                chunksize=(m.group("chunksize") or "Unknown"),
                run_mode=(m.group("run_mode") or "Unknown"),
                encoding=(m.group("encoding") or "Unknown"),
                commit_every_chunks=(m.group("commit") or "Unknown"),
                benchmark_max_level=(m.group("benchmark") or "Unknown"),
                test_up_to=(m.group("test_up_to") or "Unknown"),
                run35_compat=(m.group("run35_compat") or "Unknown"),
                max_squares=(m.group("max_squares") or "Unknown"),
                global_coverage_prune=(m.group("global_prune") or "Unknown"),
                edge_neighborhood_prune=(m.group("edge_prune") or "Unknown"),
                edge_neighborhood_lookahead=(m.group("edge_neighborhood_lookahead") or "Unknown"),
                edge_neighborhood_lookahead_node_budget=(m.group("edge_neighborhood_lookahead_node_budget") or "Unknown"),
                prioritized_edge_single_pass=(m.group("prioritized_edge_single_pass") or "Unknown"),
                check_vertex_links=(m.group("check_links") or "Unknown"),
                reuse_vertex_codes=(m.group("reuse_vertex_codes") or "Unknown"),
                fast_hash_dedup=(m.group("fast_hash_dedup") or "Unknown"),
                blob_encode_cache=(m.group("blob_encode_cache") or "Unknown"),
                stage_good_inserts=(m.group("stage_good_inserts") or "Unknown"),
                bulk_label_dedup=(m.group("bulk_label_dedup") or "Unknown"),
                frontier_sharding=(m.group("frontier_sharding") or "Unknown"),
                frontier_shards=(m.group("frontier_shards") or "Unknown"),
                frontier_sharding_min_level=(m.group("frontier_sharding_min_level") or "Unknown"),
                faceperm_by_cube=(m.group("faceperm_by_cube") or "Unknown"),
                batch_merge_inserts=(m.group("batch_merge") or "Unknown"),
            )
            runs.append(current)
            continue

        if current is None:
            continue

        if RUN_END_RE.match(raw):
            current.ended = True
            continue

        m = CHUNK_RE.match(raw)
        if m:
            lvl = int(m.group("level"))
            ls = current.level(lvl)
            ls.chunks += 1
            ls.read_ms += int(m.group("read"))
            ls.analyze_ms += int(m.group("analyze"))
            ls.reduce_ms += int(m.group("reduce"))
            ls.proc_ms += int(m.group("proc"))
            ls.generated += int(m.group("generated"))
            ls.labeled_ins += int(m.group("labeled"))
            ls.good_ins += int(m.group("good"))
            continue

        m = COMMIT_RE.match(raw)
        if m:
            lvl = int(m.group("level"))
            ls = current.level(lvl)
            ls.commit_ms += int(m.group("commit_ms"))
            continue

        m = PRUNE_RE.match(raw)
        if m:
            lvl = int(m.group("level"))
            ls = current.level(lvl)
            ls.prune_global_coverage += int(m.group("global"))
            ls.prune_edge_neighborhood += int(m.group("edge"))

    return runs


def parse_args(argv: list[str]) -> tuple[Path, int, str | None]:
    log_path = Path("run_profile.log")
    min_level = 24
    label_filter: str | None = None

    positional: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--label":
            if i + 1 >= len(argv):
                raise ValueError("missing value after --label")
            label_filter = argv[i + 1]
            i += 2
            continue
        positional.append(arg)
        i += 1

    if positional:
        log_path = Path(positional[0])
    if len(positional) > 1:
        try:
            min_level = int(positional[1])
        except ValueError as e:
            raise ValueError(f"invalid min_level: {positional[1]}") from e
    if len(positional) > 2:
        raise ValueError("too many positional args; usage: analyze_run_profile.py [log_path] [min_level] [--label LABEL]")

    return log_path, min_level, label_filter


def fmt_ms(ms: int) -> str:
    return f"{ms / 1000.0:.2f}s"


def print_run(run: RunStats, min_level: int) -> None:
    print(
        f"Run {run.index}: {run.ts} label={run.run_label} mode={run.run_mode} "
        f"n={run.n} chunksize={run.chunksize} encoding={run.encoding} commit_every={run.commit_every_chunks} "
        f"bench_max={run.benchmark_max_level} test_up_to={run.test_up_to} run35_compat={run.run35_compat} "
        f"max_squares={run.max_squares} "
        f"global_prune={run.global_coverage_prune} "
        f"edge_prune={run.edge_neighborhood_prune} edge_neighborhood_lookahead={run.edge_neighborhood_lookahead} "
        f"edge_neighborhood_lookahead_node_budget={run.edge_neighborhood_lookahead_node_budget} "
        f"prioritized_edge_single_pass={run.prioritized_edge_single_pass} "
        f"check_links={run.check_vertex_links} reuse_vertex_codes={run.reuse_vertex_codes} "
        f"fast_hash_dedup={run.fast_hash_dedup} "
        f"blob_encode_cache={run.blob_encode_cache} "
        f"stage_good_inserts={run.stage_good_inserts} "
        f"bulk_label_dedup={run.bulk_label_dedup} "
        f"frontier_sharding={run.frontier_sharding} "
        f"frontier_shards={run.frontier_shards} "
        f"frontier_sharding_min_level={run.frontier_sharding_min_level} "
        f"faceperm_by_cube={run.faceperm_by_cube} "
        f"batch_merge={run.batch_merge_inserts} ended={run.ended}"
    )
    lvls = sorted([l for l in run.levels if l >= min_level])
    if not lvls:
        print(f"  no levels >= {min_level}")
        return

    for lvl in lvls:
        ls = run.levels[lvl]
        print(
            f"  L{lvl:>2}: chunks={ls.chunks:<3} total={fmt_ms(ls.total_ms):>8} "
            f"(proc={fmt_ms(ls.proc_ms):>8}, commit={fmt_ms(ls.commit_ms):>8}) "
            f"read={fmt_ms(ls.read_ms):>8} analyze={fmt_ms(ls.analyze_ms):>8} "
            f"reduce={fmt_ms(ls.reduce_ms):>8} labeled={ls.labeled_ins} "
            f"prune_cov={ls.prune_global_coverage} prune_edge={ls.prune_edge_neighborhood}"
        )


def pct_delta(old: int, new: int) -> str:
    if old == 0:
        return "n/a"
    return f"{((new - old) * 100.0 / old):+.1f}%"


def print_compare(old: RunStats, new: RunStats, min_level: int) -> None:
    print()
    print(f"Compare run {old.index} -> run {new.index} (levels >= {min_level})")
    levels = sorted(set(old.levels) & set(new.levels))
    levels = [l for l in levels if l >= min_level]
    if not levels:
        print("  no overlapping levels in range")
        return

    for lvl in levels:
        o = old.levels.get(lvl, LevelStats())
        n = new.levels.get(lvl, LevelStats())
        print(
            f"  L{lvl:>2}: total {fmt_ms(o.total_ms)} -> {fmt_ms(n.total_ms)} ({pct_delta(o.total_ms, n.total_ms)}), "
            f"commit {fmt_ms(o.commit_ms)} -> {fmt_ms(n.commit_ms)} ({pct_delta(o.commit_ms, n.commit_ms)}), "
            f"read {fmt_ms(o.read_ms)} -> {fmt_ms(n.read_ms)} ({pct_delta(o.read_ms, n.read_ms)}), "
            f"reduce {fmt_ms(o.reduce_ms)} -> {fmt_ms(n.reduce_ms)} ({pct_delta(o.reduce_ms, n.reduce_ms)}), "
            f"prune_cov {o.prune_global_coverage} -> {n.prune_global_coverage}, "
            f"prune_edge {o.prune_edge_neighborhood} -> {n.prune_edge_neighborhood}"
        )


def main() -> int:
    try:
        log_path, min_level, label_filter = parse_args(sys.argv[1:])
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if not log_path.exists():
        print(f"missing log file: {log_path}", file=sys.stderr)
        return 1

    runs = parse_log(log_path)
    if label_filter is not None:
        runs = [r for r in runs if r.run_label == label_filter]
    if not runs:
        print("no runs found")
        return 0

    last = runs[-1]
    print_run(last, min_level=min_level)

    if len(runs) >= 2:
        prev = runs[-2]
        print()
        print_run(prev, min_level=min_level)
        print_compare(prev, last, min_level=min_level)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
