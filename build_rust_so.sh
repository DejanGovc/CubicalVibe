#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$ROOT_DIR/rust_code"
SRC_SO="$RUST_DIR/target/release/librust_code.so"
DST_SO="$ROOT_DIR/rust_code.so"

echo "[build] cargo build --release"
(cd "$RUST_DIR" && cargo build --release)

echo "[sync] copying $SRC_SO -> $DST_SO"
cp "$SRC_SO" "$DST_SO"

echo "[verify] sha256"
sha256sum "$SRC_SO" "$DST_SO"
