# CubicalVibe

Enumeration of closed cubical surfaces in the `n`-cube, with main focus on the `6`-cube. Vibe-coded in Python, Rust and SQLite3 using OpenAI Codex, hence the name. The project is mostly functional, but should currently be considered as **experimental and under construction**.

To compile and run the software, use:

```
./build_rust_so.sh
python3 cubical.py
```

The following is an AI-generated summary of the project.

---

The project uses:
- Python for orchestration and precomputation
- a Rust extension for the main search, canonical labeling, pruning, and SQLite-backed storage

## Status

This is research code. The current `n=6` pipeline is stable enough for serious runs and has successfully reached `42` squares.

The default workflow is the standard one:
- merged final output in `cplxs{n}.db`
- no good-output sharding unless explicitly enabled

There are also more advanced experimental modes for deeper runs, including:
- frontier sharding
- deeper edge-neighborhood lookahead
- sharded final good-output storage

## Repository Layout

- `cubical.py`: main entry point
- `functions.py`: Python-side combinatorial helpers and Rust precompute initialization
- `user_interface.py`: interactive parameter selection
- `build_rust_so.sh`: builds the Rust extension and copies it to `rust_code.so`
- `rust_code/`: Rust crate implementing the heavy computation
- `analyze_run_profile.py`: helper for analyzing `run_profile.log`

## Requirements

- Python 3
- Rust toolchain with `cargo`

The Python standard library is otherwise sufficient. Rust dependencies are managed through `Cargo.toml`.

## Build

Build the Rust extension with:

```bash
./build_rust_so.sh
```

This produces:

```text
rust_code.so
```

## Running

Default run:

```bash
python3 cubical.py
```

The program is interactive by default. For a non-interactive overwrite of an existing output database, use:

```bash
AUTO_OVERWRITE_DB=1 python3 cubical.py
```

Some commonly used environment variables:

- `MAX_SQUARES`
- `EDGE_NEIGHBORHOOD_LOOKAHEAD`
- `FRONTIER_SHARDS`
- `RUN_LABEL`
- `RUN_PROFILE_LOG`
- `AUTO_OVERWRITE_DB`

Example:

```bash
RUN_LABEL=max42_k4_s16 MAX_SQUARES=42 python3 cubical.py
```

## Output

By default, results are written to:

```text
cplxs{n}.db
```

with final good complexes stored in the `goodcplxs` table.

Advanced mode:

- if `GOOD_SHARDING=1` and `GOOD_FINAL_MERGE=0`, final good output is left in sharded files
- these files are named:

```text
cplxs{n}_good_part_00.db
cplxs{n}_good_part_01.db
...
```

## Logging

Runs append profiling information to:

```text
run_profile.log
```

Disable profiling for a run with:

```bash
RUN_PROFILE_LOG=0 python3 cubical.py
```

## Notes

- The code currently assumes a fresh run from the beginning; continuation from partial frontier state is not part of the current workflow.
- Overwrite is a destructive action by design: if you want to keep an old result, copy it elsewhere first.
- The optional legacy conversion helper has not yet been updated to the newest sharded-good-output workflow.

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
