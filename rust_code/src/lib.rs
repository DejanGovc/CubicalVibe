use pyo3::{PyResult,Python,prelude::*};
use std::collections::HashSet;
use rustc_hash::{FxBuildHasher, FxHashMap};
use rustc_hash::FxHasher;
use lazy_static::lazy_static;
use std::sync::{Mutex,Arc,OnceLock,RwLock};
use std::hash::{Hash, Hasher};
use pyo3::exceptions::PyRuntimeError;
use rayon::prelude::*;
use rusqlite::{params, Connection};
use chrono::Local;
use std::io::{self, Write};
use rusqlite::Error as RusqliteError;

use std::process;
use ctrlc;
use std::fs::{remove_file, rename};
use std::path::Path;
use std::time::Instant;
use std::fs::OpenOptions;
use std::env;

const COMMIT_EVERY_CHUNKS: usize = 16;
const BENCHMARK_MAX_LEVEL: Option<usize> = None;
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum RunMode {
    Safe,
    Fast,
}
const RUN_MODE: RunMode = RunMode::Safe;
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum TempEncodingMode {
    Compressed,
    Raw32,
    Adaptive,
    Combinadic,
}
const TEMP_ENCODING_MODE: TempEncodingMode = TempEncodingMode::Combinadic;
const ADAPTIVE_SPARSE_MAX_NONZERO: usize = 20;
const TAG_SPARSE: u8 = 0xA1;
const TAG_RAW32: u8 = 0xA2;
const TAG_COMBINADIC: u8 = 0xA3;
const TAG_COMP_FALLBACK: u8 = 0xA4;
const ULTRA_MAX_K: usize = 240;
const ULTRA_MAX_N: usize = 240;
static BATCH_MERGE_INSERTS: OnceLock<bool> = OnceLock::new();
static CHECK_VERTEX_LINKS: OnceLock<bool> = OnceLock::new();
static BLOB_ENCODE_CACHE: OnceLock<bool> = OnceLock::new();
static STAGE_GOOD_INSERTS: OnceLock<bool> = OnceLock::new();
static BULK_LABEL_DEDUP: OnceLock<bool> = OnceLock::new();
static FRONTIER_SHARDING: OnceLock<bool> = OnceLock::new();
static FRONTIER_DIRECT_DEDUP: OnceLock<bool> = OnceLock::new();
static FRONTIER_CANONICAL_REPRESENTATIVE: OnceLock<bool> = OnceLock::new();
static FRONTIER_SHARDS: OnceLock<usize> = OnceLock::new();
static FRONTIER_SHARDING_MIN_LEVEL: OnceLock<usize> = OnceLock::new();
static RUN35_COMPAT: OnceLock<bool> = OnceLock::new();
static REUSE_VERTEX_CODES: OnceLock<bool> = OnceLock::new();
static FAST_HASH_DEDUP: OnceLock<bool> = OnceLock::new();
static TEST_UP_TO: OnceLock<Option<usize>> = OnceLock::new();
static RUN_PROFILE_LOG: OnceLock<bool> = OnceLock::new();
static GOOD_SHARDING: OnceLock<bool> = OnceLock::new();
static GOOD_SHARDS: OnceLock<usize> = OnceLock::new();
static GOOD_FINAL_MERGE: OnceLock<bool> = OnceLock::new();
static FINALIZE_CHECKPOINT_LABELS: OnceLock<bool> = OnceLock::new();
static CTRLC_HANDLER_INSTALLED: OnceLock<()> = OnceLock::new();

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) => {
            let x = v.trim().to_ascii_lowercase();
            x == "1" || x == "true" || x == "on" || x == "yes"
        }
        Err(_) => default,
    }
}

fn env_bool_negated(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) => {
            let x = v.trim().to_ascii_lowercase();
            !(x == "0" || x == "false" || x == "off" || x == "no")
        }
        Err(_) => default,
    }
}

fn run35_compat_enabled() -> bool {
    *RUN35_COMPAT.get_or_init(|| env_bool("RUN35_COMPAT", false))
}

fn run_profile_log_enabled() -> bool {
    *RUN_PROFILE_LOG.get_or_init(|| env_bool("RUN_PROFILE_LOG", true))
}

fn good_sharding_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *GOOD_SHARDING.get_or_init(|| env_bool("GOOD_SHARDING", false))
}

fn good_shards() -> usize {
    if run35_compat_enabled() {
        return 1;
    }
    *GOOD_SHARDS.get_or_init(|| match env::var("GOOD_SHARDS") {
        Ok(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(16),
        Err(_) => 16,
    })
}

fn good_final_merge_enabled() -> bool {
    if run35_compat_enabled() {
        return true;
    }
    *GOOD_FINAL_MERGE.get_or_init(|| env_bool_negated("GOOD_FINAL_MERGE", false))
}

fn finalize_checkpoint_labels_enabled() -> bool {
    if run35_compat_enabled() {
        return true;
    }
    *FINALIZE_CHECKPOINT_LABELS.get_or_init(|| env_bool("FINALIZE_CHECKPOINT_LABELS", false))
}

fn batch_merge_inserts_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *BATCH_MERGE_INSERTS.get_or_init(|| {
        env_bool("BATCH_MERGE_INSERTS", false)
    })
}

fn check_vertex_links_enabled() -> bool {
    if run35_compat_enabled() {
        return true;
    }
    *CHECK_VERTEX_LINKS.get_or_init(|| {
        env_bool_negated("CHECK_VERTEX_LINKS", true)
    })
}

fn blob_encode_cache_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *BLOB_ENCODE_CACHE.get_or_init(|| {
        env_bool_negated("BLOB_ENCODE_CACHE", false)
    })
}

fn stage_good_inserts_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *STAGE_GOOD_INSERTS.get_or_init(|| {
        env_bool_negated("STAGE_GOOD_INSERTS", true)
    })
}

fn bulk_label_dedup_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *BULK_LABEL_DEDUP.get_or_init(|| {
        env_bool("BULK_LABEL_DEDUP", false)
    })
}

fn frontier_sharding_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *FRONTIER_SHARDING.get_or_init(|| env_bool("FRONTIER_SHARDING", true))
}

fn frontier_direct_dedup_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *FRONTIER_DIRECT_DEDUP.get_or_init(|| env_bool("FRONTIER_DIRECT_DEDUP", false))
}

fn frontier_canonical_representative_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *FRONTIER_CANONICAL_REPRESENTATIVE
        .get_or_init(|| env_bool("FRONTIER_CANONICAL_REPRESENTATIVE", false))
}

fn frontier_shards() -> usize {
    if run35_compat_enabled() {
        return 1;
    }
    *FRONTIER_SHARDS.get_or_init(|| match env::var("FRONTIER_SHARDS") {
        Ok(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(16),
        Err(_) => 16,
    })
}

fn frontier_sharding_min_level() -> usize {
    if run35_compat_enabled() {
        return usize::MAX;
    }
    *FRONTIER_SHARDING_MIN_LEVEL.get_or_init(|| match env::var("FRONTIER_SHARDING_MIN_LEVEL") {
        Ok(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 1)
            .unwrap_or(2),
        Err(_) => 2,
    })
}

fn reuse_vertex_codes_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *REUSE_VERTEX_CODES.get_or_init(|| env_bool_negated("REUSE_VERTEX_CODES", true))
}

fn fast_hash_dedup_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *FAST_HASH_DEDUP.get_or_init(|| env_bool_negated("FAST_HASH_DEDUP", true))
}

fn test_up_to_level() -> Option<usize> {
    *TEST_UP_TO.get_or_init(|| {
        match env::var("TEST_UP_TO") {
            Ok(v) => {
                let x = v.trim().to_ascii_lowercase();
                if x.is_empty()
                    || x == "false"
                    || x == "off"
                    || x == "no"
                    || x == "none"
                    || x == "inf"
                    || x == "infinity"
                    || x == "all"
                    || x == "full"
                    || x == "max"
                {
                    None
                } else {
                    x.parse::<usize>().ok().filter(|n| *n > 0)
                }
            }
            Err(_) => None,
        }
    })
}

fn test_up_to_for_log() -> String {
    match test_up_to_level() {
        Some(v) => v.to_string(),
        None => "None".to_string(),
    }
}

enum DedupSet<T: Eq + Hash> {
    Std(HashSet<T>),
    Fx(HashSet<T, FxBuildHasher>),
}

impl<T: Eq + Hash> DedupSet<T> {
    fn with_capacity(cap: usize, fast: bool) -> Self {
        if fast {
            DedupSet::Fx(HashSet::with_capacity_and_hasher(cap, FxBuildHasher::default()))
        } else {
            DedupSet::Std(HashSet::with_capacity(cap))
        }
    }

    fn insert(&mut self, val: T) -> bool {
        match self {
            DedupSet::Std(s) => s.insert(val),
            DedupSet::Fx(s) => s.insert(val),
        }
    }
}

struct ChunkProfile {
    analyze_ms: u128,
    reduce_ms: u128,
    generated: usize,
    inserted_labeled: usize,
    inserted_good: usize,
}

enum LabelInsertMode<'a> {
    Single(rusqlite::Statement<'a>),
    Sharded(Vec<rusqlite::Statement<'a>>),
}

struct SourceCursor<B: BitBackend> {
    conn: Connection,
    path: String,
    last_id: i64,
    buf: Vec<(i64, B)>,
    pos: usize,
    exhausted: bool,
}

fn tempcplx_path(level: usize) -> String {
    format!("tempcplxs{}.db", level)
}

fn tempcplx_shard_path(level: usize, shard: usize) -> String {
    format!("tempcplxs{}_s{:02}.db", level, shard)
}

fn templabel_shard_path(shard: usize) -> String {
    format!("templabels_s{:02}.db", shard)
}

fn tempgood_shard_path(shard: usize) -> String {
    format!("tempgood_s{:02}.db", shard)
}

fn db_stem(db_name: &str) -> String {
    db_name.strip_suffix(".db").unwrap_or(db_name).to_string()
}

fn goodcplx_shard_path(db_stem: &str, shard: usize) -> String {
    format!("{}_good_part_{:02}.db", db_stem, shard)
}

fn frontier_input_paths(level: usize) -> Vec<String> {
    if frontier_sharding_enabled() {
        let mut paths = Vec::new();
        for shard in 0..frontier_shards() {
            let path = tempcplx_shard_path(level, shard);
            if Path::new(&path).exists() {
                paths.push(path);
            }
        }
        if !paths.is_empty() {
            return paths;
        }
    }
    vec![tempcplx_path(level)]
}

fn shard_for_label<B: BitBackend>(label: &B, shards: usize) -> usize {
    let mut hasher = FxHasher::default();
    label.hash(&mut hasher);
    (hasher.finish() as usize) % shards
}

fn initialize_good_shards_from_main(
    conn: &Connection,
    num_good_shards: usize,
    db_stem: &str,
) -> rusqlite::Result<()> {
    let mut shard_conns: Vec<Connection> = Vec::with_capacity(num_good_shards);
    for shard in 0..num_good_shards {
        let shard_path = goodcplx_shard_path(db_stem, shard);
        let shard_conn = Connection::open(&shard_path)?;
        tune_sqlite_main(&shard_conn)?;
        shard_conn.execute(
            "CREATE TABLE IF NOT EXISTS goodcplxs (
                cplx BLOB PRIMARY KEY,
                id INTEGER
            )",
            [],
        )?;
        shard_conns.push(shard_conn);
    }

    let mut stmt = conn.prepare("SELECT cplx, id FROM goodcplxs ORDER BY id")?;
    let rows = stmt.query_map([], |row| {
        let cplx_blob: Vec<u8> = row.get(0)?;
        let id: i64 = row.get(1)?;
        Ok((cplx_blob, id))
    })?;

    let mut insert_stmts: Vec<rusqlite::Statement<'_>> = Vec::with_capacity(num_good_shards);
    for shard_conn in shard_conns.iter_mut() {
        insert_stmts.push(
            shard_conn.prepare("INSERT OR IGNORE INTO goodcplxs (cplx, id) VALUES (?, ?)")?
        );
    }

    for row in rows {
        let (cplx_blob, id) = row?;
        let cplx = blob_to_int(&cplx_blob);
        let shard = shard_for_label(&cplx, num_good_shards);
        insert_stmts[shard].execute(params![cplx_blob, id])?;
    }

    drop(insert_stmts);
    drop(stmt);
    drop(shard_conns);
    Ok(())
}

fn finalize_good_shards(num_good_shards: usize, db_stem: &str) -> rusqlite::Result<(usize, FinalizePhaseTimings)> {
    let total_start = Instant::now();
    let mut inserted_total = 0usize;
    let mut timings = FinalizePhaseTimings::default();
    for shard in 0..num_good_shards {
        let stage_path = tempgood_shard_path(shard);
        if !Path::new(&stage_path).exists() {
            continue;
        }
        let checkpoint_start = Instant::now();
        checkpoint_temp_db(&stage_path)?;
        timings.checkpoint_ms += checkpoint_start.elapsed().as_millis();
        let good_path = goodcplx_shard_path(db_stem, shard);
        let setup_start = Instant::now();
        let good_conn = Connection::open(&good_path)?;
        tune_sqlite_main(&good_conn)?;
        good_conn.execute(
            "CREATE TABLE IF NOT EXISTS goodcplxs (
                cplx BLOB PRIMARY KEY,
                id INTEGER
            )",
            [],
        )?;
        timings.setup_ms += setup_start.elapsed().as_millis();
        let attach_start = Instant::now();
        good_conn.execute("ATTACH DATABASE ? AS stagedgood", params![stage_path])?;
        timings.attach_ms += attach_start.elapsed().as_millis();
        let dedup_start = Instant::now();
        good_conn.execute(
            "CREATE TEMP TABLE dedup_good_ids AS
             SELECT MIN(id) AS id
             FROM stagedgood.staged_good
             GROUP BY cplx",
            [],
        )?;
        timings.dedup_ms += dedup_start.elapsed().as_millis();
        let insert_start = Instant::now();
        let inserted = good_conn.execute(
            "INSERT OR IGNORE INTO goodcplxs (cplx, id)
             SELECT s.cplx, s.id
             FROM stagedgood.staged_good AS s
             JOIN dedup_good_ids AS d ON s.id = d.id
             ORDER BY s.id",
            [],
        )?;
        timings.insert_ms += insert_start.elapsed().as_millis();
        inserted_total += inserted;
        let cleanup_start = Instant::now();
        good_conn.execute("DROP TABLE dedup_good_ids", [])?;
        good_conn.execute("DETACH DATABASE stagedgood", [])?;
        drop(good_conn);
        remove_sqlite_artifacts(&stage_path);
        timings.cleanup_ms += cleanup_start.elapsed().as_millis();
    }
    timings.total_ms = total_start.elapsed().as_millis();
    Ok((inserted_total, timings))
}

fn merge_good_shards_into_main(conn: &Connection, num_good_shards: usize, db_stem: &str) -> rusqlite::Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS goodcplxs (
            cplx BLOB PRIMARY KEY,
            id INTEGER
        )",
        [],
    )?;
    for shard in 0..num_good_shards {
        let good_path = goodcplx_shard_path(db_stem, shard);
        if !Path::new(&good_path).exists() {
            continue;
        }
        conn.execute("ATTACH DATABASE ? AS goodshard", params![good_path])?;
        conn.execute(
            "INSERT OR IGNORE INTO goodcplxs (cplx, id)
             SELECT cplx, id FROM goodshard.goodcplxs ORDER BY id",
            [],
        )?;
        conn.execute("DETACH DATABASE goodshard", [])?;
    }
    Ok(())
}

fn fetch_source_batch<B: BitBackend>(
    conn: &Connection,
    level: usize,
    last_id: i64,
    limit: usize,
    holder: &DataHolder<B>,
) -> Result<Vec<(i64, B)>, RusqliteError> {
    let select_sql = format!(
        "SELECT id, cplx FROM cplxs{} WHERE id > ?1 ORDER BY id LIMIT ?2",
        level
    );
    let mut stmt = conn.prepare(&select_sql)?;
    let rows_iter = stmt.query_map(params![last_id, limit as i64], |row| {
        let id: i64 = row.get(0)?;
        let cplx_blob: Vec<u8> = row.get(1)?;
        Ok((id, decode_temp_bits(&cplx_blob, holder)))
    })?;
    rows_iter.collect()
}

pub mod functions;
mod u256;
use u256::{BitBackend, U256, U768};

fn u256_to_bits<B: BitBackend>(val: &U256) -> B {
    debug_assert!(B::FITS_U256);
    B::from_u256(*val)
}

fn bits_to_u256<B: BitBackend>(val: &B) -> U256 {
    debug_assert!(B::FITS_U256);
    val.to_u256()
}

fn encode_temp_bits<B: BitBackend>(val: &B, holder: &DataHolder<B>) -> Vec<u8> {
    if B::FITS_U256 {
        encode_temp_blob(&bits_to_u256(val), *holder.n2)
    } else {
        encode_temp_blob_wide(val, *holder.n2, &holder.ultra_binom)
    }
}

fn encode_main_bits<B: BitBackend>(val: &B, holder: &DataHolder<B>) -> Vec<u8> {
    if B::FITS_U256 {
        int_to_blob(&bits_to_u256(val))
    } else {
        int_to_blob_wide(val, active_square_bytes(*holder.n2))
    }
}

fn decode_temp_bits<B: BitBackend>(blob: &[u8], holder: &DataHolder<B>) -> B {
    if B::FITS_U256 {
        u256_to_bits(&decode_temp_blob(blob, *holder.n2))
    } else {
        decode_temp_blob_wide(blob, *holder.n2, &holder.ultra_binom)
    }
}

// RUSQLITE ERROR TO PYTHON ERROR:

fn convert_rusqlite_error<T>(result: Result<T, RusqliteError>) -> PyResult<T> {
    result.map_err(|e| PyRuntimeError::new_err(format!("Database operation failed: {}", e)))
}

fn env_i64(name: &str, default: i64) -> i64 {
    match env::var(name) {
        Ok(v) => v.trim().parse::<i64>().ok().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    match env::var(name) {
        Ok(v) => v.trim().parse::<u64>().ok().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(v) => v.trim().parse::<usize>().ok().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_sync(name: &str, default: &str) -> String {
    match env::var(name) {
        Ok(v) => {
            let x = v.trim().to_ascii_uppercase();
            match x.as_str() {
                "OFF" | "NORMAL" | "FULL" | "EXTRA" => x,
                _ => default.to_string(),
            }
        }
        Err(_) => default.to_string(),
    }
}

fn main_sync_setting() -> String {
    if run35_compat_enabled() {
        return "NORMAL".to_string();
    }
    let default = match RUN_MODE {
        RunMode::Safe => "NORMAL",
        RunMode::Fast => "NORMAL",
    };
    env_sync("MAIN_SYNCHRONOUS", default)
}

fn temp_sync_setting() -> String {
    if run35_compat_enabled() {
        return "NORMAL".to_string();
    }
    let default = match RUN_MODE {
        RunMode::Safe => "NORMAL",
        RunMode::Fast => "OFF",
    };
    env_sync("TEMP_SYNCHRONOUS", default)
}

fn checkpoint_mode() -> Option<String> {
    if run35_compat_enabled() {
        return Some("PASSIVE".to_string());
    }
    match env::var("WAL_CHECKPOINT_MODE") {
        Ok(v) => {
            let x = v.trim().to_ascii_uppercase();
            match x.as_str() {
                "NONE" | "OFF" | "NO" => None,
                "PASSIVE" | "FULL" | "RESTART" | "TRUNCATE" => Some(x),
                _ => Some("PASSIVE".to_string()),
            }
        }
        Err(_) => Some("PASSIVE".to_string()),
    }
}

fn checkpoint_every_levels() -> usize {
    if run35_compat_enabled() {
        return 1;
    }
    env_usize("CHECKPOINT_EVERY_LEVELS", 1).max(1)
}

fn tune_sqlite_main(conn: &Connection) -> rusqlite::Result<()> {
    let sync = main_sync_setting();
    let cache_size = env_i64("MAIN_CACHE_SIZE", -100000);
    let mmap_size = env_u64("MAIN_MMAP_SIZE", 268435456);
    let wal_autockpt = env_u64("MAIN_WAL_AUTOCHECKPOINT", 100000);
    let pragma = format!(
        "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = {sync};
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = {cache_size};
            PRAGMA mmap_size = {mmap_size};
            PRAGMA wal_autocheckpoint = {wal_autockpt};
        "
    );
    conn.execute_batch(&pragma)
}

fn tune_sqlite_temp(conn: &Connection) -> rusqlite::Result<()> {
    let sync = temp_sync_setting();
    let cache_size = env_i64(
        "TEMP_CACHE_SIZE",
        match RUN_MODE {
            RunMode::Safe => -100000,
            RunMode::Fast => -200000,
        },
    );
    let mmap_size = env_u64(
        "TEMP_MMAP_SIZE",
        match RUN_MODE {
            RunMode::Safe => 268435456,
            RunMode::Fast => 1073741824,
        },
    );
    let wal_autockpt = env_u64(
        "TEMP_WAL_AUTOCHECKPOINT",
        match RUN_MODE {
            RunMode::Safe => 0,
            RunMode::Fast => 0,
        },
    );
    let pragma = format!(
        "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = {sync};
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = {cache_size};
            PRAGMA mmap_size = {mmap_size};
            PRAGMA wal_autocheckpoint = {wal_autockpt};
        "
    );
    conn.execute_batch(&pragma)
}

fn tune_sqlite_finalize_output(conn: &Connection) -> rusqlite::Result<()> {
    let cache_size = env_i64(
        "FINALIZE_CACHE_SIZE",
        match RUN_MODE {
            RunMode::Safe => -100000,
            RunMode::Fast => -200000,
        },
    );
    let mmap_size = env_u64(
        "FINALIZE_MMAP_SIZE",
        match RUN_MODE {
            RunMode::Safe => 268435456,
            RunMode::Fast => 1073741824,
        },
    );
    let pragma = format!(
        "
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = {cache_size};
            PRAGMA mmap_size = {mmap_size};
        "
    );
    conn.execute_batch(&pragma)
}

fn checkpoint_temp_db(path: &str) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    tune_sqlite_temp(&conn)?;
    let _: (i64, i64, i64) = conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?;
    Ok(())
}

fn file_size(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn remove_sqlite_artifacts(db_path: &str) {
    for suffix in ["", "-wal", "-shm"] {
        let pathstring = format!("{}{}", db_path, suffix);
        let path = Path::new(&pathstring);
        if path.exists() {
            let _ = remove_file(path);
        }
    }
}

#[derive(Default, Clone, Copy)]
struct FinalizePhaseTimings {
    setup_ms: u128,
    checkpoint_ms: u128,
    attach_ms: u128,
    dedup_ms: u128,
    insert_ms: u128,
    rename_ms: u128,
    cleanup_ms: u128,
    total_ms: u128,
}

fn rename_sqlite_artifacts(from_path: &str, to_path: &str) -> rusqlite::Result<()> {
    remove_sqlite_artifacts(to_path);
    for suffix in ["", "-wal", "-shm"] {
        let from = format!("{from_path}{suffix}");
        let to = format!("{to_path}{suffix}");
        let path = Path::new(&from);
        if path.exists() {
            rename(path, Path::new(&to))
                .map_err(|e| RusqliteError::ToSqlConversionFailure(Box::new(e)))?;
        }
    }
    Ok(())
}

// GLOBAL VARIABLES AND PRECOMPUTED FUNCTIONS

pub struct DataHolder<B: BitBackend = U256> {
    n_dim: Arc<usize>,
    n2: Arc<usize>,
    n1: Arc<usize>,
    n0: Arc<usize>,
    chunksize: Arc<usize>,
    boundaries: Arc<Vec<B>>,
    bboundaries: Arc<Vec<B>>,
    edgeboundaries: Arc<Vec<B>>,
    nnbhd: Arc<Vec<Vec<usize>>>,
    permlist_cache: Arc<RwLock<FxHashMap<usize, Arc<Vec<Vec<usize>>>>>>,
    cubes: Arc<Vec<Vec<usize>>>,
    squares2: Arc<Vec<Vec<usize>>>,
    square_index: Arc<FxHashMap<Vec<usize>, usize>>,
    faceperm_cache: Arc<RwLock<FxHashMap<(usize, Vec<usize>), Arc<Vec<usize>>>>>,
    edgesquares: Arc<Vec<B>>,
    cycles3: Arc<Vec<Vec<B>>>,
    cycles4: Arc<Vec<Vec<B>>>,
    cycles5: Arc<Vec<Vec<B>>>,
    ultra_binom: Arc<Vec<Vec<B>>>,
}

impl<B: BitBackend> DataHolder<B> {
    pub fn new(
        n_dim: usize,
        n2: usize,
        n1: usize,
        n0: usize,
        chunksize: usize,
        boundaries: Vec<B>,
        bboundaries: Vec<B>,
        edgeboundaries: Vec<B>,
        nnbhd: Vec<Vec<usize>>,
        cubes: Vec<Vec<usize>>,
        edgesquares: Vec<B>,
        cycles3: Vec<Vec<B>>,
        cycles4: Vec<Vec<B>>,
        cycles5: Vec<Vec<B>>,
        ultra_binom: Vec<Vec<B>>,
    ) -> Self {
        let squares2 = generate_cubes_of_dim(n_dim, 2);
        let mut square_index = FxHashMap::default();
        for (idx, cube) in squares2.iter().enumerate() {
            square_index.insert(cube.clone(), idx);
        }
        DataHolder {
            n_dim: Arc::new(n_dim),
            n2: Arc::new(n2),
            n1: Arc::new(n1),
            n0: Arc::new(n0),
            chunksize: Arc::new(chunksize),
            boundaries: Arc::new(boundaries),
            bboundaries: Arc::new(bboundaries),
            edgeboundaries: Arc::new(edgeboundaries),
            nnbhd: Arc::new(nnbhd),
            permlist_cache: Arc::new(RwLock::new(FxHashMap::default())),
            cubes: Arc::new(cubes),
            squares2: Arc::new(squares2),
            square_index: Arc::new(square_index),
            faceperm_cache: Arc::new(RwLock::new(FxHashMap::default())),
            edgesquares: Arc::new(edgesquares),
            cycles3: Arc::new(cycles3),
            cycles4: Arc::new(cycles4),
            cycles5: Arc::new(cycles5),
            ultra_binom: Arc::new(ultra_binom),
        }
    }

    pub fn get_faceperm_for(&self, cube_idx: usize, tau: &[usize]) -> Arc<Vec<usize>> {
        let key = (cube_idx, tau.to_vec());
        if let Some(existing) = self.faceperm_cache.read().unwrap().get(&key).cloned() {
            return existing;
        }
        let root = &self.cubes[cube_idx];
        let mut perm = Vec::with_capacity(self.squares2.len());
        for square in self.squares2.iter() {
            let renamed = rename_square(square, root, tau, *self.n_dim);
            perm.push(
                *self.square_index
                    .get(&renamed)
                    .expect("Failure."),
            );
        }
        let computed = Arc::new(perm);
        let mut cache = self.faceperm_cache.write().unwrap();
        cache.entry(key).or_insert_with(|| computed.clone()).clone()
    }

    pub fn get_permlist(&self, encoded: usize, arity: usize) -> Arc<Vec<Vec<usize>>> {
        if let Some(existing) = self
            .permlist_cache
            .read()
            .unwrap()
            .get(&encoded)
            .cloned()
        {
            return existing;
        }
        let computed = Arc::new(build_permlist_for_pattern(encoded, arity));
        let mut cache = self.permlist_cache.write().unwrap();
        cache.entry(encoded).or_insert_with(|| computed.clone()).clone()
    }
}

fn decode_octal_pattern(mut encoded: usize, arity: usize) -> Vec<usize> {
    let mut digits = vec![0usize; arity];
    for i in (0..arity).rev() {
        digits[i] = encoded & 0b111;
        encoded >>= 3;
    }
    digits
}

fn generate_cubes_of_dim(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn rec(pos: usize, n: usize, twos_left: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if pos == n {
            if twos_left == 0 {
                out.push(cur.clone());
            }
            return;
        }
        let remaining = n - pos;
        if twos_left > remaining {
            return;
        }
        for value in 0..=2usize {
            let new_twos_left = if value == 2 {
                if twos_left == 0 {
                    continue;
                }
                twos_left - 1
            } else {
                twos_left
            };
            cur.push(value);
            rec(pos + 1, n, new_twos_left, cur, out);
            cur.pop();
        }
    }

    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(n);
    rec(0, n, k, &mut cur, &mut out);
    out
}

fn rename_square(square: &[usize], root: &[usize], tau: &[usize], n_dim: usize) -> Vec<usize> {
    let mut flipped = Vec::with_capacity(n_dim);
    for i in 0..n_dim {
        let v = if root[i] == 1 && square[i] <= 1 {
            1 - square[i]
        } else {
            square[i]
        };
        flipped.push(v);
    }
    tau.iter().map(|&i| flipped[i]).collect()
}

fn sorting_perm_usize(values: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by_key(|&idx| values[idx]);
    indices
}

fn repeat_counts_sorted(values: &[usize], sorted_perm: &[usize]) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut last: Option<usize> = None;
    let mut count = 0usize;
    for &idx in sorted_perm {
        let value = values[idx];
        if last == Some(value) {
            count += 1;
        } else {
            if count > 0 {
                counts.push(count);
            }
            last = Some(value);
            count = 1;
        }
    }
    if count > 0 {
        counts.push(count);
    }
    counts
}

fn enumerate_permutations_in_place(items: &mut [usize], start: usize, out: &mut Vec<Vec<usize>>) {
    if start >= items.len() {
        out.push(items.to_vec());
        return;
    }
    for i in start..items.len() {
        items.swap(start, i);
        enumerate_permutations_in_place(items, start + 1, out);
        items.swap(start, i);
    }
}

fn permutations_of_range(start: usize, len: usize) -> Vec<Vec<usize>> {
    if len == 0 {
        return vec![Vec::new()];
    }
    let mut items: Vec<usize> = (start..start + len).collect();
    let mut out = Vec::new();
    enumerate_permutations_in_place(&mut items, 0, &mut out);
    out
}

fn build_permlist_for_pattern(encoded: usize, arity: usize) -> Vec<Vec<usize>> {
    let digits = decode_octal_pattern(encoded, arity);
    let pi = sorting_perm_usize(&digits);
    let counts = repeat_counts_sorted(&digits, &pi);

    let mut grouped: Vec<Vec<usize>> = vec![Vec::new()];
    let mut offset = 0usize;
    for count in counts {
        let block_perms = permutations_of_range(offset, count);
        let mut next = Vec::with_capacity(grouped.len() * block_perms.len());
        for prefix in &grouped {
            for perm in &block_perms {
                let mut combined = prefix.clone();
                combined.extend_from_slice(perm);
                next.push(combined);
            }
        }
        grouped = next;
        offset += count;
    }

    grouped
        .into_iter()
        .map(|tau| tau.into_iter().map(|i| pi[i]).collect())
        .collect()
}

lazy_static! {
    pub static ref N: Mutex<usize> = Mutex::new(0);
    pub static ref N2: Mutex<usize> = Mutex::new(0);
    pub static ref N1: Mutex<usize> = Mutex::new(0);
    pub static ref N0: Mutex<usize> = Mutex::new(0);
    pub static ref CHUNKSIZE: Mutex<usize> = Mutex::new(0);

    pub static ref PRECOMPUTED_BOUNDARIES: Mutex<Vec<Vec<u128>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_BBOUNDARIES: Mutex<Vec<Vec<u128>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_EDGEBOUNDARIES: Mutex<Vec<Vec<u128>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_NNBHD: Mutex<Vec<Vec<usize>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_CUBES: Mutex<Vec<Vec<usize>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_EDGESQUARES: Mutex<Vec<Vec<u128>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_CYCLES3: Mutex<Vec<Vec<Vec<u128>>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_CYCLES4: Mutex<Vec<Vec<Vec<u128>>>> = Mutex::new(vec![]);
    pub static ref PRECOMPUTED_CYCLES5: Mutex<Vec<Vec<Vec<u128>>>> = Mutex::new(vec![]);
    pub static ref ULTRA_BINOM: Vec<Vec<U256>> = build_ultra_binom::<U256>(ULTRA_MAX_N);
}

fn build_ultra_binom<B: BitBackend>(max_n: usize) -> Vec<Vec<B>> {
    let mut table = vec![vec![B::zero(); max_n + 1]; max_n + 1];
    table[0][0] = B::from(1u8);
    for n in 1..=max_n {
        table[n][0] = B::from(1u8);
        let max_k = n;
        for k in 1..=max_k {
            if k == n {
                table[n][k] = B::from(1u8);
            } else {
                let mut v = table[n - 1][k - 1];
                v += table[n - 1][k];
                table[n][k] = v;
            }
        }
    }
    table
}

#[pyfunction]
fn load_precomputed_values(_py: Python,
    n: usize,
    n2: usize,
    n1: usize,
    n0: usize,
    chunksize: usize,
    boundaries_values: Vec<Vec<u128>>,
    bboundaries_values: Vec<Vec<u128>>,    // This fits in a U256 because 2^n vertices require 3*2^n bits to encode.
    edgeboundaries_values: Vec<Vec<u128>>, // This fits in a U256 because 2^n vertices require 3*2^n bits to encode.
    nnbhd_values: Vec<Vec<usize>>,
    cubes_values: Vec<Vec<usize>>,
    edgesquares_values: Vec<Vec<u128>>,
    cycles3_values: Vec<Vec<Vec<u128>>>,
    cycles4_values: Vec<Vec<Vec<u128>>>,
    cycles5_values: Vec<Vec<Vec<u128>>>
) {
    let mut n_ref = N.lock().unwrap();
    let mut n2_ref = N2.lock().unwrap();
    let mut n1_ref = N1.lock().unwrap();
    let mut n0_ref = N0.lock().unwrap();
    let mut chunksize_ref = CHUNKSIZE.lock().unwrap();
    let mut precomputed_boundaries = PRECOMPUTED_BOUNDARIES.lock().unwrap();
    let mut precomputed_bboundaries = PRECOMPUTED_BBOUNDARIES.lock().unwrap();
    let mut precomputed_edgeboundaries = PRECOMPUTED_EDGEBOUNDARIES.lock().unwrap();
    let mut precomputed_nnbhd = PRECOMPUTED_NNBHD.lock().unwrap();
    let mut precomputed_cubes = PRECOMPUTED_CUBES.lock().unwrap();
    let mut precomputed_edgesquares = PRECOMPUTED_EDGESQUARES.lock().unwrap();
    let mut precomputed_cycles3 = PRECOMPUTED_CYCLES3.lock().unwrap();
    let mut precomputed_cycles4 = PRECOMPUTED_CYCLES4.lock().unwrap();
    let mut precomputed_cycles5 = PRECOMPUTED_CYCLES5.lock().unwrap();

    *n_ref = n;
    *n2_ref = n2;
    *n1_ref = n1;
    *n0_ref = n0;
    *chunksize_ref = chunksize;
    *precomputed_boundaries = boundaries_values;
    *precomputed_bboundaries = bboundaries_values;
    *precomputed_edgeboundaries = edgeboundaries_values;
    *precomputed_nnbhd = nnbhd_values;
    *precomputed_cubes = cubes_values;
    *precomputed_edgesquares = edgesquares_values;
    *precomputed_cycles3 = cycles3_values;
    *precomputed_cycles4 = cycles4_values;
    *precomputed_cycles5 = cycles5_values;
}

fn initialize_holder<B: BitBackend>() -> DataHolder<B> {
    let n2 = *N2.lock().unwrap();
    let n1 = *N1.lock().unwrap();
    let n0 = *N0.lock().unwrap();
    let n_dim = *N.lock().unwrap();
    let chunksize = *CHUNKSIZE.lock().unwrap();
    let boundaries = PRECOMPUTED_BOUNDARIES
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|v| B::from_u128_slice(&v))
        .collect();
    let bboundaries = PRECOMPUTED_BBOUNDARIES
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|v| B::from_u128_slice(&v))
        .collect();
    let edgeboundaries = PRECOMPUTED_EDGEBOUNDARIES
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|v| B::from_u128_slice(&v))
        .collect();
    let nnbhd = PRECOMPUTED_NNBHD.lock().unwrap().clone();
    let cubes = PRECOMPUTED_CUBES.lock().unwrap().clone();
    let edgesquares = PRECOMPUTED_EDGESQUARES
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|v| B::from_u128_slice(&v))
        .collect();
    let cycles3 = PRECOMPUTED_CYCLES3
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|outer_vec| {
            outer_vec
                .into_iter()
                .map(|v| B::from_u128_slice(&v))
                .collect()
        })
        .collect();
    let cycles4 = PRECOMPUTED_CYCLES4
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|outer_vec| {
            outer_vec
                .into_iter()
                .map(|v| B::from_u128_slice(&v))
                .collect()
        })
        .collect();
    let cycles5 = PRECOMPUTED_CYCLES5
        .lock()
        .unwrap()
        .clone()
        .into_iter()
        .map(|outer_vec| {
            outer_vec
                .into_iter()
                .map(|v| B::from_u128_slice(&v))
                .collect()
        })
        .collect();
    let holder = DataHolder::new(
        n_dim,
        n2,
        n1,
        n0,
        chunksize,
        boundaries,
        bboundaries,
        edgeboundaries,
        nnbhd,
        cubes,
        edgesquares,
        cycles3,
        cycles4,
        cycles5,
        build_ultra_binom::<B>(n2),
    );
    return holder
}

// DOING THINGS DIRECTLY IN RUST:

// DIFFERENT CONVERSION ALGORITHM:

fn diffcompress(vec: &Vec<u8>) -> Vec<u8> {
    let mut vvec = vec.clone();
    let mut newvec = vec![];
    let pos0 = vvec.pop().expect("Fail.");
    let pos1 = vvec.pop().expect("Fail.");
    let pos2 = vvec.pop().expect("Fail.");
    let pos3 = vvec.pop().expect("Fail.");
    let mut u256input = U256::zero();
    let mut index = 0;
    for v in vvec.iter() {
        u256input += U256::from(*v)<<index;
        index += 8;
    }

    index = 0;
    let mut count = 0;
    let mut u256output = U256::zero();
    while u256input != U256::zero() {
        let n = u256input.trailing_zeros() as usize;
        u256output += U256::from((n+1) as u8)<<index; // Adding 1 because 0 can be ambiguous.
        count += 1;
        u256input >>= n+1;
        index += 4;
    }
    
    for _ in 0..(count+1)/2 { // +1 to behave as the ceiling function.
        newvec.push((u256output & U256::from(0b11111111)).to_usize().expect("Fail.") as u8);
        u256output >>= 8;
    }
    newvec.push(pos3);
    newvec.push(pos2);
    newvec.push(pos1);
    newvec.push(pos0);
    newvec
}

fn diffdecompress(cprsd: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut u256output = U256::zero();

    let mut compressed = cprsd.to_vec();
    let pos0 = compressed.pop().expect("Fail.");
    let pos1 = compressed.pop().expect("Fail.");
    let pos2 = compressed.pop().expect("Fail.");
    let pos3 = compressed.pop().expect("Fail.");

    let mut index = 0;

    for pv in compressed {
        index += (pv & 15)-1; // Substracting 1 because it is added in compression.
        u256output += U256::one()<<(index as usize);
        index += 1;
        if (pv>>4) & 15 == 0 {
            break;
        }
        index += ((pv>>4) & 15)-1; // Substracting 1 because it is added in compression.
        u256output += U256::one()<<(index as usize);
        index += 1;
    }

    for _ in 0..32 {
        let byte = (u256output & U256::from(255)).to_usize().expect("Fail.") as u8;
        u256output >>= 8;
        if byte != 0u8 {
            output.push(byte);
        }
    }

    output.push(pos3);
    output.push(pos2);
    output.push(pos1);
    output.push(pos0);

    output
}

fn int_to_blob(val: &U256) -> Vec<u8> {
    let mut vval = val.clone();
    let mask = U256::from(0b11111111u128);
    let mut nonzero = vec![];
    let mut pos0 = 0u8;
    let mut pos1 = 0u8;
    let mut pos2 = 0u8;
    let mut pos3 = 0u8;
    for i in 0..8 {
        if &mask & vval != U256::zero() {
            pos0 += 1<<i;
            nonzero.push((&mask & vval).to_usize().expect("Fail.") as u8);
        }
        vval >>= 8;
    }
    for i in 0..8 {
        if &mask & vval != U256::zero() {
            pos1 += 1<<i;
            nonzero.push((&mask & vval).to_usize().expect("Fail.") as u8);
        }
        vval >>= 8;
    }
    for i in 0..8 {
        if &mask & vval != U256::zero() {
            pos2 += 1<<i;
            nonzero.push((&mask & vval).to_usize().expect("Fail.") as u8);
        }
        vval >>= 8;
    }
    for i in 0..8 {
        if &mask & vval != U256::zero() {
            pos3 += 1<<i;
            nonzero.push((&mask & vval).to_usize().expect("Fail.") as u8);
        }
        vval >>= 8;
    }
    nonzero.push(pos3);
    nonzero.push(pos2);
    nonzero.push(pos1);
    nonzero.push(pos0);
    diffcompress(&nonzero)
}

fn blob_to_int(blob: &[u8]) -> U256 {
    if let Some(&1) = blob.last() {
        if blob.iter().take(blob.len() - 1).all(|&x| x == 0) {
            return U256::one();
        }
    }
    let mut bblob = diffdecompress(blob);
    let pos0 = bblob.pop().expect("Fail.") as u32;
    let pos1 = bblob.pop().expect("Fail.") as u32;
    let pos2 = bblob.pop().expect("Fail.") as u32;
    let pos3 = bblob.pop().expect("Fail.") as u32;
    let mut pos = 0u32;
    pos += pos0 + (pos1<<8) + (pos2<<16) + (pos3<<24);
    //println!("Positions: {},{},{},{}->{}",pos0,pos1,pos2,pos3,pos);
    let mut val = U256::zero();
    let mut index = 0;
    for i in 0..32 {
        if pos & (1<<i as u32) != 0u32 {
            val += U256::from(bblob[index])<<(8*i);
            index += 1;
        }
    }
    //println!("blob: {:?}, val: {}, pos = {}",blob,val,pos);
    val
}

fn int_to_blob_raw32(val: &U256) -> Vec<u8> {
    val.to_be_bytes_vec()
}

fn blob_to_int_raw32(blob: &[u8]) -> U256 {
    U256::from_be_slice(blob)
}

fn int_to_blob_combinadic(val: &U256, n2: usize) -> Option<Vec<u8>> {
    if n2 > ULTRA_MAX_N {
        return None;
    }
    let mut bits = *val;
    let mut k = 0usize;
    let mut rank = U256::zero();
    while bits != U256::zero() {
        let p = bits.trailing_zeros() as usize;
        if p >= n2 {
            break;
        }
        k += 1;
        if k > ULTRA_MAX_K {
            return None;
        }
        rank += ULTRA_BINOM[p][k];
        bits ^= U256::one() << p;
    }

    let rank_arr = rank.to_be_bytes_vec();
    let first_nz = rank_arr.iter().position(|&b| b != 0).unwrap_or(rank_arr.len());
    let rank_bytes = &rank_arr[first_nz..];
    let mut out = Vec::with_capacity(3 + rank_bytes.len());
    out.push(TAG_COMBINADIC);
    out.push(k as u8);
    out.push(rank_bytes.len() as u8);
    out.extend_from_slice(rank_bytes);
    Some(out)
}

fn blob_to_int_combinadic(blob: &[u8], n2: usize) -> Option<U256> {
    if blob.len() < 3 || n2 > ULTRA_MAX_N {
        return None;
    }
    let k = blob[1] as usize;
    if k > ULTRA_MAX_K {
        return None;
    }
    let len = blob[2] as usize;
    if blob.len() != 3 + len {
        return None;
    }
    if len > 32 {
        return None;
    }
    let mut padded = [0u8; 32];
    padded[(32 - len)..].copy_from_slice(&blob[3..]);
    let mut rank = U256::from_be_slice(&padded);
    let mut positions = vec![0usize; k];
    let mut x = n2;
    for j in (1..=k).rev() {
        if x == 0 {
            return None;
        }
        let mut cand = x - 1;
        while ULTRA_BINOM[cand][j] > rank {
            if cand == 0 {
                return None;
            }
            cand -= 1;
        }
        rank -= &ULTRA_BINOM[cand][j];
        positions[j - 1] = cand;
        x = cand;
    }

    let mut out = U256::zero();
    for p in positions {
        out |= U256::one() << p;
    }
    Some(out)
}

fn int_to_blob_sparse(val: &U256) -> Vec<u8> {
    let mut vval = *val;
    let mask = U256::from(0b11111111u128);
    let mut pos: u32 = 0;
    let mut payload: Vec<u8> = Vec::new();

    for i in 0..32 {
        let byte = (&mask & vval).to_usize().expect("Fail.") as u8;
        if byte != 0 {
            pos |= 1u32 << i;
            payload.push(byte);
        }
        vval >>= 8;
    }

    let mut out = Vec::with_capacity(1 + 4 + payload.len());
    out.push(TAG_SPARSE);
    out.extend_from_slice(&pos.to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

fn blob_to_int_sparse(blob: &[u8]) -> U256 {
    if blob.len() < 5 {
        return U256::zero();
    }
    let pos = u32::from_le_bytes([blob[1], blob[2], blob[3], blob[4]]);
    let mut index = 5usize;
    let mut val = U256::zero();
    for i in 0..32 {
        if pos & (1u32 << i) != 0 {
            if index >= blob.len() {
                break;
            }
            val += U256::from(blob[index]) << (8 * i);
            index += 1;
        }
    }
    val
}

fn nonzero_byte_count(val: &U256) -> usize {
    let mut count = 0usize;
    let mut vval = *val;
    let mask = U256::from(0b11111111u128);
    for _ in 0..32 {
        if (&mask & vval) != U256::zero() {
            count += 1;
        }
        vval >>= 8;
    }
    count
}

fn encode_temp_blob(val: &U256, n2: usize) -> Vec<u8> {
    match TEMP_ENCODING_MODE {
        TempEncodingMode::Compressed => int_to_blob(val),
        TempEncodingMode::Raw32 => int_to_blob_raw32(val),
        TempEncodingMode::Adaptive => {
            let nnz = nonzero_byte_count(val);
            if nnz <= ADAPTIVE_SPARSE_MAX_NONZERO {
                int_to_blob_sparse(val)
            } else {
                let mut out = Vec::with_capacity(1 + 32);
                out.push(TAG_RAW32);
                out.extend_from_slice(&val.to_be_bytes_vec());
                out
            }
        }
        TempEncodingMode::Combinadic => {
            if let Some(b) = int_to_blob_combinadic(val, n2) {
                b
            } else {
                let mut out = Vec::new();
                out.push(TAG_COMP_FALLBACK);
                out.extend_from_slice(&int_to_blob(val));
                out
            }
        }
    }
}

fn decode_temp_blob(blob: &[u8], n2: usize) -> U256 {
    match TEMP_ENCODING_MODE {
        TempEncodingMode::Compressed => blob_to_int(blob),
        TempEncodingMode::Raw32 => blob_to_int_raw32(blob),
        TempEncodingMode::Adaptive => {
            if blob.is_empty() {
                return U256::zero();
            }
            match blob[0] {
                TAG_SPARSE => blob_to_int_sparse(blob),
                TAG_RAW32 => blob_to_int_raw32(&blob[1..]),
                _ => blob_to_int(blob),
            }
        }
        TempEncodingMode::Combinadic => {
            if blob.is_empty() {
                return U256::zero();
            }
            match blob[0] {
                TAG_COMBINADIC => blob_to_int_combinadic(blob, n2).unwrap_or_else(U256::zero),
                TAG_COMP_FALLBACK => blob_to_int(&blob[1..]),
                _ => blob_to_int(blob),
            }
        }
    }
}

fn main_mask_bytes(byte_width: usize) -> usize {
    (byte_width + 7) / 8
}

fn active_square_bytes(n2: usize) -> usize {
    (n2 + 7) / 8
}

fn int_to_blob_wide<B: BitBackend>(val: &B, byte_width: usize) -> Vec<u8> {
    let pos_bytes = main_mask_bytes(byte_width);
    let be = val.to_be_bytes_vec();
    let mut payload = Vec::new();
    let mut pos = vec![0u8; pos_bytes];
    for (i, &byte) in be[be.len() - byte_width..].iter().rev().enumerate() {
        if byte != 0 {
            pos[i / 8] |= 1u8 << (i % 8);
            payload.push(byte);
        }
    }
    payload.extend_from_slice(&pos);
    payload
}

fn blob_to_int_wide<B: BitBackend>(blob: &[u8], byte_width: usize) -> B {
    let pos_bytes = main_mask_bytes(byte_width);
    if blob.len() < pos_bytes {
        return B::zero();
    }
    let split = blob.len() - pos_bytes;
    let payload = &blob[..split];
    let pos = &blob[split..];
    let full_width = B::BIT_WIDTH / 8;
    let mut be = vec![0u8; full_width];
    let mut payload_index = 0usize;
    for i in 0..byte_width {
        if ((pos[i / 8] >> (i % 8)) & 1u8) != 0 {
            if payload_index >= payload.len() {
                break;
            }
            be[full_width - 1 - i] = payload[payload_index];
            payload_index += 1;
        }
    }
    B::from_be_slice(&be)
}

fn int_to_blob_raw_wide<B: BitBackend>(val: &B) -> Vec<u8> {
    val.to_be_bytes_vec()
}

fn blob_to_int_raw_wide<B: BitBackend>(blob: &[u8]) -> B {
    B::from_be_slice(blob)
}

fn int_to_blob_sparse_wide<B: BitBackend>(val: &B, byte_width: usize) -> Vec<u8> {
    let pos_bytes = main_mask_bytes(byte_width);
    let be = val.to_be_bytes_vec();
    let mut payload = Vec::new();
    let mut out = Vec::with_capacity(1 + pos_bytes + byte_width);
    out.push(TAG_SPARSE);
    let mut pos = vec![0u8; pos_bytes];
    for (i, &byte) in be[be.len() - byte_width..].iter().rev().enumerate() {
        if byte != 0 {
            pos[i / 8] |= 1u8 << (i % 8);
            payload.push(byte);
        }
    }
    out.extend_from_slice(&pos);
    out.extend_from_slice(&payload);
    out
}

fn blob_to_int_sparse_wide<B: BitBackend>(blob: &[u8], byte_width: usize) -> B {
    let pos_bytes = main_mask_bytes(byte_width);
    if blob.len() < 1 + pos_bytes {
        return B::zero();
    }
    let pos = &blob[1..1 + pos_bytes];
    let payload = &blob[1 + pos_bytes..];
    let full_width = B::BIT_WIDTH / 8;
    let mut be = vec![0u8; full_width];
    let mut payload_index = 0usize;
    for i in 0..byte_width {
        if ((pos[i / 8] >> (i % 8)) & 1u8) != 0 {
            if payload_index >= payload.len() {
                break;
            }
            be[full_width - 1 - i] = payload[payload_index];
            payload_index += 1;
        }
    }
    B::from_be_slice(&be)
}

fn nonzero_byte_count_wide<B: BitBackend>(val: &B, byte_width: usize) -> usize {
    let be = val.to_be_bytes_vec();
    be[be.len() - byte_width..]
        .iter()
        .filter(|&&b| b != 0)
        .count()
}

fn int_to_blob_combinadic_wide<B: BitBackend>(val: &B, n2: usize, ultra_binom: &[Vec<B>]) -> Option<Vec<u8>> {
    let mut bits = *val;
    let mut k = 0usize;
    let mut rank = B::zero();
    while bits != B::zero() {
        let p = bits.trailing_zeros() as usize;
        if p >= n2 {
            break;
        }
        k += 1;
        rank += ultra_binom[p][k];
        bits ^= B::one() << p;
    }
    let rank_arr = rank.to_be_bytes_vec();
    let first_nz = rank_arr.iter().position(|&b| b != 0).unwrap_or(rank_arr.len());
    let rank_bytes = &rank_arr[first_nz..];
    let mut out = Vec::with_capacity(3 + rank_bytes.len());
    out.push(TAG_COMBINADIC);
    out.push(k as u8);
    out.push(rank_bytes.len() as u8);
    out.extend_from_slice(rank_bytes);
    Some(out)
}

fn blob_to_int_combinadic_wide<B: BitBackend>(blob: &[u8], n2: usize, ultra_binom: &[Vec<B>]) -> Option<B> {
    if blob.len() < 3 {
        return None;
    }
    let k = blob[1] as usize;
    let len = blob[2] as usize;
    if blob.len() != 3 + len {
        return None;
    }
    let byte_width = B::BIT_WIDTH / 8;
    if len > byte_width {
        return None;
    }
    let mut padded = vec![0u8; byte_width];
    padded[(byte_width - len)..].copy_from_slice(&blob[3..]);
    let mut rank = B::from_be_slice(&padded);
    let mut positions = vec![0usize; k];
    let mut x = n2;
    for j in (1..=k).rev() {
        if x == 0 {
            return None;
        }
        let mut cand = x - 1;
        while ultra_binom[cand][j] > rank {
            if cand == 0 {
                return None;
            }
            cand -= 1;
        }
        rank -= ultra_binom[cand][j];
        positions[j - 1] = cand;
        x = cand;
    }
    let mut out = B::zero();
    for p in positions {
        out |= B::one() << p;
    }
    Some(out)
}

fn encode_temp_blob_wide<B: BitBackend>(val: &B, n2: usize, ultra_binom: &[Vec<B>]) -> Vec<u8> {
    let byte_width = active_square_bytes(n2);
    match TEMP_ENCODING_MODE {
        TempEncodingMode::Compressed => int_to_blob_wide(val, byte_width),
        TempEncodingMode::Raw32 => int_to_blob_raw_wide(val),
        TempEncodingMode::Adaptive => {
            let nnz = nonzero_byte_count_wide(val, byte_width);
            if nnz <= ADAPTIVE_SPARSE_MAX_NONZERO {
                int_to_blob_sparse_wide(val, byte_width)
            } else {
                let be = val.to_be_bytes_vec();
                let mut out = Vec::with_capacity(1 + byte_width);
                out.push(TAG_RAW32);
                out.extend_from_slice(&be[be.len() - byte_width..]);
                out
            }
        }
        TempEncodingMode::Combinadic => {
            if let Some(b) = int_to_blob_combinadic_wide(val, n2, ultra_binom) {
                b
            } else {
                let mut out = Vec::new();
                out.push(TAG_COMP_FALLBACK);
                out.extend_from_slice(&int_to_blob_wide(val, byte_width));
                out
            }
        }
    }
}

fn decode_temp_blob_wide<B: BitBackend>(blob: &[u8], n2: usize, ultra_binom: &[Vec<B>]) -> B {
    let active_bytes = active_square_bytes(n2);
    match TEMP_ENCODING_MODE {
        TempEncodingMode::Compressed => blob_to_int_wide(blob, active_bytes),
        TempEncodingMode::Raw32 => blob_to_int_raw_wide(blob),
        TempEncodingMode::Adaptive => {
            if blob.is_empty() {
                return B::zero();
            }
            match blob[0] {
                TAG_SPARSE => blob_to_int_sparse_wide(blob, active_bytes),
                TAG_RAW32 => blob_to_int_raw_wide(&blob[1..]),
                _ => blob_to_int_wide(blob, active_bytes),
            }
        }
        TempEncodingMode::Combinadic => {
            if blob.is_empty() {
                return B::zero();
            }
            match blob[0] {
                TAG_COMBINADIC => blob_to_int_combinadic_wide(blob, n2, ultra_binom).unwrap_or_else(B::zero),
                TAG_COMP_FALLBACK => blob_to_int_wide(&blob[1..], active_bytes),
                _ => {
                    if blob.len() == active_bytes {
                        B::from_be_slice(blob)
                    } else {
                        blob_to_int_wide(blob, active_bytes)
                    }
                }
            }
        }
    }
}

fn process_chunk<B: BitBackend>(
    label_mode: &mut LabelInsertMode<'_>,
    stmt2: Option<&mut rusqlite::Statement<'_>>,
    stage_stmt1: Option<&mut rusqlite::Statement<'_>>,
    stage_stmt2: Option<&mut rusqlite::Statement<'_>>,
    stage_good_stmt: Option<&mut rusqlite::Statement<'_>>,
    good_shard_stmts: Option<&mut Vec<rusqlite::Statement<'_>>>,
    holder: &DataHolder<B>,
    chunk: &[B],
    nall: &mut usize,
    ngood: &mut usize,
    dbprefix: &String,
    use_batch_merge_inserts: bool,
    use_blob_encode_cache: bool,
    use_stage_good_inserts: bool,
    use_good_sharding: bool,
    use_bulk_label_dedup: bool,
    use_frontier_sharding: bool,
    use_frontier_direct_dedup: bool,
    use_frontier_canonical_representative: bool,
    num_frontier_shards: usize,
    num_good_shards: usize,
) -> rusqlite::Result<ChunkProfile> {
    enum AnalyzeMode {
        DbOrB,
        D,
        Plain,
    }
    let extend_fun: fn(&B, &DataHolder<B>) -> Vec<B> = if dbprefix == "db" {
        functions::disconnected_withbdry_extendonce
    } else if dbprefix == "b" {
        functions::withbdry_extendonce
    } else if dbprefix == "d" {
        functions::disconnected_extendonce
    } else if dbprefix == "" {
        functions::anotherextendonce // THIS WILL SECRETLY BE THE N3 EXTEND (!)
    } else {
        println!("FAILED.");
        process::exit(0)
    };
    let mode = if dbprefix == "db" || dbprefix == "b" {
        AnalyzeMode::DbOrB
    } else if dbprefix == "d" {
        AnalyzeMode::D
    } else {
        AnalyzeMode::Plain
    };
    let check_links = check_vertex_links_enabled();
    let reuse_vertex_codes = reuse_vertex_codes_enabled();
    let analyze_start = Instant::now();
    let (generated, analyzed): (usize, Vec<(B, B, bool, bool)>) = chunk
        .into_par_iter()
        .fold(
            || (0usize, Vec::new()),
            |mut acc, c| {
            let candidates = extend_fun(c, &holder);
            acc.0 += candidates.len();
            for candidate in candidates {
                match mode {
                    AnalyzeMode::DbOrB => {
                        let (b0, _) = functions::testedges(&candidate, &holder);
                        if !b0 {
                            continue;
                        }
                        let (bad, reallybad, label) = if check_links {
                            if reuse_vertex_codes {
                                let dd = functions::vertex_codes(&candidate, &holder);
                                let (bad, reallybad) = functions::bad_and_reallybad_from_codes(&candidate, &dd, &holder);
                                let label = functions::cubicalcanlabel_from_codes(&candidate, &dd, &holder);
                                (bad, reallybad, label)
                            } else {
                                let (bad, reallybad) = functions::bad_and_reallybad(&candidate, &holder);
                                let label = functions::cubicalcanlabel(&candidate, &holder);
                                (bad, reallybad, label)
                            }
                        } else {
                            (false, false, functions::cubicalcanlabel(&candidate, &holder))
                        };
                        if reallybad {
                            continue;
                        }
                        // let label = functions::nolabel(&candidate, &holder);
                        acc.1.push((label, candidate, true, !bad));
                    }
                    AnalyzeMode::D => {
                        let (b0, b1) = functions::testedges(&candidate, &holder);
                        if !b0 {
                            continue;
                        }
                        let (bad, reallybad, label) = if check_links {
                            if reuse_vertex_codes {
                                let dd = functions::vertex_codes(&candidate, &holder);
                                let (bad, reallybad) = functions::bad_and_reallybad_from_codes(&candidate, &dd, &holder);
                                let label = functions::cubicalcanlabel_from_codes(&candidate, &dd, &holder);
                                (bad, reallybad, label)
                            } else {
                                let (bad, reallybad) = functions::bad_and_reallybad(&candidate, &holder);
                                let label = functions::cubicalcanlabel(&candidate, &holder);
                                (bad, reallybad, label)
                            }
                        } else {
                            (false, false, functions::cubicalcanlabel(&candidate, &holder))
                        };
                        if reallybad {
                            continue;
                        }
                        // let label = functions::nolabel(&candidate, &holder);
                        acc.1.push((label, candidate, true, b1 && !bad));
                    }
                    AnalyzeMode::Plain => {
                        let (b0, b1) = functions::testedges(&candidate, &holder);
                        if !b0 {
                            continue;
                        }
                        let (bad, reallybad, label) = if check_links {
                            if reuse_vertex_codes {
                                let dd = functions::vertex_codes(&candidate, &holder);
                                let (bad, reallybad) = functions::bad_and_reallybad_from_codes(&candidate, &dd, &holder);
                                let label = functions::cubicalcanlabel_from_codes(&candidate, &dd, &holder);
                                (bad, reallybad, label)
                            } else {
                                let (bad, reallybad) = functions::bad_and_reallybad(&candidate, &holder);
                                let label = functions::cubicalcanlabel(&candidate, &holder);
                                (bad, reallybad, label)
                            }
                        } else {
                            (false, false, functions::cubicalcanlabel(&candidate, &holder))
                        };
                        if reallybad {
                            continue;
                        }
                        let keep_labeled = !b1;
                        let keep_good = b1 && !bad;
                        if !(keep_labeled || keep_good) {
                            continue;
                        }
                        // let label = functions::nolabel(&candidate, &holder);
                        acc.1.push((label, candidate, keep_labeled, keep_good));
                    }
                }
            }
            acc
            },
        )
        .reduce(
            || (0usize, Vec::new()),
            |mut a, mut b| {
                a.0 += b.0;
                a.1.append(&mut b.1);
                a
            },
        );
    let analyze_ms = analyze_start.elapsed().as_millis();

    print!("Analyzed({}ms). ", analyze_ms);
    io::stdout().flush().unwrap();

    let insert_start = Instant::now();
    let mut inserted_labeled = 0usize;
    let mut inserted_good = 0usize;
    // Chunk-local dedup to avoid redundant INSERT OR IGNORE work.
    // This preserves first-occurrence order of each canonical label.
    let use_fast_hash_dedup = fast_hash_dedup_enabled();
    let mut seen_labeled = DedupSet::with_capacity(analyzed.len(), use_fast_hash_dedup);
    let mut seen_good = DedupSet::with_capacity(analyzed.len() / 8 + 1, use_fast_hash_dedup);
    let mut label_blob_cache: FxHashMap<B, Vec<u8>> = FxHashMap::default();
    let mut cplx_blob_cache: FxHashMap<B, Vec<u8>> = FxHashMap::default();
    let mut good_blob_cache: FxHashMap<B, Vec<u8>> = FxHashMap::default();
    if use_batch_merge_inserts {
        let stage1 = stage_stmt1.expect("Batch mode requires stage stmt1.");
        let stage2 = stage_stmt2.expect("Batch mode requires stage stmt2.");
        for (val1, val2, keep_labeled, keep_good) in analyzed.iter() {
            if *keep_labeled && seen_labeled.insert(*val1) {
                *nall += 1;
                let frontier_rep = if use_frontier_canonical_representative { *val1 } else { *val2 };
                if !(use_frontier_sharding && use_frontier_direct_dedup) {
                    inserted_labeled += 1;
                }
                if use_blob_encode_cache {
                    let cplx_blob = cplx_blob_cache
                        .entry(frontier_rep)
                        .or_insert_with(|| encode_temp_bits(&frontier_rep, holder));
                    if use_frontier_canonical_representative {
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, &*cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, &*cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            stage1.execute(params![&*cplx_blob, *nall])?;
                        }
                    } else {
                        let label_blob = label_blob_cache
                            .entry(*val1)
                            .or_insert_with(|| encode_temp_bits(val1, holder));
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, &*label_blob, &*cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, &*label_blob, &*cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            stage1.execute(params![&*label_blob, *nall, &*cplx_blob])?;
                        }
                    }
                } else {
                    let cplx_blob = encode_temp_bits(&frontier_rep, holder);
                    if use_frontier_canonical_representative {
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            stage1.execute(params![cplx_blob, *nall])?;
                        }
                    } else {
                        let label_blob = encode_temp_bits(val1, holder);
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, label_blob, cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, label_blob, cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            stage1.execute(params![label_blob, *nall, cplx_blob])?;
                        }
                    }
                }
            }
            if *keep_good && seen_good.insert(*val1) {
                *ngood += 1;
                inserted_good += 1;
                if use_blob_encode_cache {
                        let good_blob = good_blob_cache
                            .entry(*val1)
                            .or_insert_with(|| encode_main_bits(val1, holder));
                    stage2.execute(params![&*good_blob, *ngood])?;
                } else {
                    stage2.execute(
                        params![encode_main_bits(val1, holder), *ngood]
                    )?;
                }
            }
        }
    } else {
        let mut good_stage_opt = stage_good_stmt;
        let mut good_shards_opt = good_shard_stmts;
        let mut good_stmt_opt = stmt2;
        for (val1, val2, keep_labeled, keep_good) in analyzed.iter() {
            if *keep_labeled && seen_labeled.insert(*val1) {
                *nall += 1;
                let frontier_rep = if use_frontier_canonical_representative { *val1 } else { *val2 };
                if !(use_frontier_sharding && use_frontier_direct_dedup) {
                    inserted_labeled += 1;
                }
                // `nall` becomes the append-order sequence in bulk dedup mode.
                if use_blob_encode_cache {
                    let cplx_blob = cplx_blob_cache
                        .entry(frontier_rep)
                        .or_insert_with(|| encode_temp_bits(&frontier_rep, holder));
                    if use_frontier_canonical_representative {
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, &*cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, &*cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![&*cplx_blob, *nall])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("non-bulk label path uses single label mode"),
                            }
                        }
                    } else {
                        let label_blob = label_blob_cache
                            .entry(*val1)
                            .or_insert_with(|| encode_temp_bits(val1, holder));
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, &*label_blob, &*cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, &*label_blob, &*cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![&*label_blob, *nall, &*cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("non-bulk label path uses single label mode"),
                            }
                        }
                    }
                } else {
                    let cplx_blob = encode_temp_bits(&frontier_rep, holder);
                    if use_frontier_canonical_representative {
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![cplx_blob, *nall])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("non-bulk label path uses single label mode"),
                            }
                        }
                    } else {
                        let label_blob = encode_temp_bits(val1, holder);
                        if use_frontier_sharding {
                            let shard = shard_for_label(val1, num_frontier_shards);
                            match label_mode {
                                LabelInsertMode::Sharded(stmts) => {
                                    let affected =
                                        stmts[shard].execute(params![*nall, label_blob, cplx_blob])?;
                                    if use_frontier_direct_dedup {
                                        inserted_labeled += affected;
                                    }
                                }
                                LabelInsertMode::Single(_) => unreachable!("frontier sharding requires sharded label mode"),
                            }
                        } else if use_bulk_label_dedup {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![*nall, label_blob, cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("bulk label dedup without sharding uses single label mode"),
                            }
                        } else {
                            match label_mode {
                                LabelInsertMode::Single(stmt1) => {
                                    stmt1.execute(params![label_blob, *nall, cplx_blob])?;
                                }
                                LabelInsertMode::Sharded(_) => unreachable!("non-bulk label path uses single label mode"),
                            }
                        }
                    }
                }
            }
            if *keep_good && seen_good.insert(*val1) {
                *ngood += 1;
                inserted_good += 1;
                if use_good_sharding {
                    let good_shards = good_shards_opt
                        .as_deref_mut()
                        .expect("Good sharding requires shard statements.");
                    let shard = shard_for_label(val1, num_good_shards);
                    if use_blob_encode_cache {
                        let good_blob = good_blob_cache
                            .entry(*val1)
                            .or_insert_with(|| encode_main_bits(val1, holder));
                        good_shards[shard].execute(params![*ngood, &*good_blob])?;
                    } else {
                        good_shards[shard].execute(params![*ngood, encode_main_bits(val1, holder)])?;
                    }
                } else if use_stage_good_inserts {
                    let good_stage = good_stage_opt
                        .as_deref_mut()
                        .expect("Good stage mode requires stage_good stmt.");
                    if use_blob_encode_cache {
                        let good_blob = good_blob_cache
                            .entry(*val1)
                            .or_insert_with(|| encode_main_bits(val1, holder));
                        good_stage.execute(params![&*good_blob, *ngood])?;
                    } else {
                        good_stage.execute(params![encode_main_bits(val1, holder), *ngood])?;
                    }
                } else if use_blob_encode_cache {
                    let good_blob = good_blob_cache
                        .entry(*val1)
                        .or_insert_with(|| encode_main_bits(val1, holder));
                    good_stmt_opt
                        .as_deref_mut()
                        .expect("Direct good insert mode requires stmt2.")
                        .execute(params![&*good_blob, *ngood])?;
                } else {
                    good_stmt_opt
                        .as_deref_mut()
                        .expect("Direct good insert mode requires stmt2.")
                        .execute(params![encode_main_bits(val1, holder), *ngood])?;
                }
            }
        }
    }
    let insert_ms = insert_start.elapsed().as_millis();
    print!("Reduced({}ms).\r", insert_ms);
    io::stdout().flush().unwrap();
    Ok(ChunkProfile {
        analyze_ms,
        reduce_ms: insert_ms,
        generated,
        inserted_labeled,
        inserted_good,
    })
}

#[pyfunction]
fn main_loop(_py: Python, imin: usize, imax: usize, ngood: usize, initial_lencplxs: usize, initial_lengoodcplxs: usize, dbprefix: String, db_name: String) -> PyResult<()> {
    convert_rusqlite_error(do_main_loop(imin,imax,ngood,initial_lencplxs,initial_lengoodcplxs,&dbprefix,&db_name))
}

#[pyfunction]
fn test(_py: Python) -> PyResult<()> {
    // Placeholder for testing purposes.
    Ok(())
}

fn do_main_loop(imin: usize, imax: usize, ngood: usize, initial_lencplxs: usize, initial_lengoodcplxs: usize, dbprefix: &String, db_name: &String) -> rusqlite::Result<()> {
    let n_value = *N.lock().unwrap();
    match n_value {
        0..=5 => do_main_loop_typed::<u128>(imin, imax, ngood, initial_lencplxs, initial_lengoodcplxs, dbprefix, db_name),
        6 => do_main_loop_typed::<U256>(imin, imax, ngood, initial_lencplxs, initial_lengoodcplxs, dbprefix, db_name),
        7 => do_main_loop_typed::<U768>(imin, imax, ngood, initial_lencplxs, initial_lengoodcplxs, dbprefix, db_name),
        n => Err(RusqliteError::InvalidParameterName(format!(
            "unsupported cube dimension {n}; compute backends are prepared only for n <= 7"
        ))),
    }
}

fn do_main_loop_typed<B: BitBackend>(imin: usize, imax: usize, mut ngood: usize, initial_lencplxs: usize, initial_lengoodcplxs: usize, dbprefix: &String, db_name: &String) -> rusqlite::Result<()> {
    // let vec = vec![16, 1, 3, 19, 51,0,0,0,0];

    // println!("{:?}",diffcompress(&vec));
    // println!("{:?}",diffdecompress(&diffcompress(&vec)));
    // println!("{:?}",diffcompress(&diffdecompress(&vec)));

    CTRLC_HANDLER_INSTALLED.get_or_init(|| {
        ctrlc::set_handler(move || {
            println!("\nReceived SIGINT, terminating.");
            process::exit(0);
        }).expect("Error setting Ctrl-C handler");
    });

    let holder = initialize_holder::<B>();
    let conn = Connection::open(db_name)?;
    tune_sqlite_main(&conn)?;
    let mut profile_log = if run_profile_log_enabled() {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("run_profile.log")
            .ok()
    } else {
        None
    };
    if let Some(log) = profile_log.as_mut() {
        let run_label = env::var("RUN_LABEL").unwrap_or_else(|_| "-".to_string());
        let n_value = *N.lock().unwrap();
        let _ = writeln!(
            log,
            "{} run_start run_label={} n={} dbprefix={} chunksize={} backend_limbs={} backend_bits={} run_mode={:?} temp_encoding_mode={:?} commit_every_chunks={} benchmark_max_level={:?} test_up_to={} run35_compat={} max_squares={} global_coverage_prune={} edge_neighborhood_prune={} edge_neighborhood_lookahead={} edge_neighborhood_lookahead_node_budget={} prioritized_edge_single_pass={} canon_three_hop_refinement={} check_vertex_links={} reuse_vertex_codes={} fast_hash_dedup={} blob_encode_cache={} stage_good_inserts={} good_sharding={} good_shards={} good_final_merge={} bulk_label_dedup={} frontier_sharding={} frontier_shards={} frontier_sharding_min_level={} frontier_direct_dedup={} frontier_canonical_representative={} finalize_checkpoint_labels={} faceperm_cache={} batch_merge_inserts={} main_sync={} temp_sync={} main_wal_autocheckpoint={} temp_wal_autocheckpoint={} checkpoint_mode={} checkpoint_every_levels={}",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            run_label,
            n_value,
            dbprefix,
            *holder.chunksize,
            B::LIMBS,
            B::BIT_WIDTH,
            RUN_MODE,
            TEMP_ENCODING_MODE,
            COMMIT_EVERY_CHUNKS,
            BENCHMARK_MAX_LEVEL,
            test_up_to_for_log(),
            run35_compat_enabled(),
            functions::max_squares()
                .map(|x| x.to_string())
                .unwrap_or_else(|| "None".to_string()),
            functions::global_coverage_prune_enabled(),
            functions::edge_neighborhood_prune_enabled(),
            functions::edge_neighborhood_lookahead_depth(),
            functions::edge_neighborhood_lookahead_node_budget(),
            functions::prioritized_edge_single_pass_enabled(),
            functions::canon_three_hop_refinement_enabled(),
            check_vertex_links_enabled(),
            reuse_vertex_codes_enabled(),
            fast_hash_dedup_enabled(),
            blob_encode_cache_enabled(),
            stage_good_inserts_enabled(),
            good_sharding_enabled(),
            good_shards(),
            good_final_merge_enabled(),
            bulk_label_dedup_enabled(),
            frontier_sharding_enabled(),
            frontier_shards(),
            frontier_sharding_min_level(),
            frontier_direct_dedup_enabled(),
            frontier_canonical_representative_enabled(),
            finalize_checkpoint_labels_enabled(),
            "on_demand",
            batch_merge_inserts_enabled(),
            main_sync_setting(),
            temp_sync_setting(),
            env_u64("MAIN_WAL_AUTOCHECKPOINT", 100000),
            env_u64(
                "TEMP_WAL_AUTOCHECKPOINT",
                match RUN_MODE {
                    RunMode::Safe => 0,
                    RunMode::Fast => 0,
                }
            ),
            checkpoint_mode().unwrap_or_else(|| "NONE".to_string()),
            checkpoint_every_levels()
        );
    }
    let mut lencplxs = initial_lencplxs;
    let mut lengoodcplxs = initial_lengoodcplxs;
    let good_db_stem = db_stem(db_name);
    let use_good_sharding = good_sharding_enabled();
    let num_good_shards = if use_good_sharding { good_shards() } else { 1 };
    if use_good_sharding {
        initialize_good_shards_from_main(&conn, num_good_shards, &good_db_stem)?;
    }
    let use_closed_mode_for_square_cap = dbprefix.is_empty() || dbprefix == "d";
    let square_cap_stop = if use_closed_mode_for_square_cap {
        functions::max_squares()
            .map(|x| std::cmp::min(imax, x as usize))
            .unwrap_or(imax)
    } else {
        imax
    };
    let benchmark_stop = BENCHMARK_MAX_LEVEL.map_or(square_cap_stop, |m| std::cmp::min(m, square_cap_stop));
    if benchmark_stop < imax {
        println!("Benchmark mode: stopping at level {} (requested imax={}).", benchmark_stop, imax);
    }
    let level_stop = if let Some(t) = test_up_to_level() {
        let test_stop = std::cmp::min(t, imax);
        if test_stop < benchmark_stop {
            println!("Test mode: stopping at level {} (benchmark_stop={}).", test_stop, benchmark_stop);
        }
        std::cmp::min(benchmark_stop, test_stop)
    } else {
        benchmark_stop
    };
    let lookahead_depth = functions::edge_neighborhood_lookahead_depth();
    if dbprefix.is_empty() && lookahead_depth >= 4 {
        println!(
            "WARNING: EDGE_NEIGHBORHOOD_LOOKAHEAD={} may fail if EDGE_NEIGHBORHOOD_LOOKAHEAD_NODE_BUDGET={} is exceeded.",
            lookahead_depth,
            functions::edge_neighborhood_lookahead_node_budget()
        );
    }
    println!("{}: # squares: {}; # cplxs: {}; # goodcplxs: {}", Local::now().format("%Y-%m-%d %H:%M:%S"), 1, initial_lencplxs, initial_lengoodcplxs);
    for i in imin..level_stop+1 {
        functions::reset_extend_prune_counters();
        if i == 65 && !matches!(TEMP_ENCODING_MODE, TempEncodingMode::Combinadic) {
            println!("WARNING: current blob codec is not designed for >64-square support.");
        }
        let use_frontier_sharding = frontier_sharding_enabled() && i >= frontier_sharding_min_level();
        let use_frontier_direct_dedup = use_frontier_sharding && frontier_direct_dedup_enabled();
        let source_paths = frontier_input_paths(i - 1);
        let mut goodtxconn_opt = if use_good_sharding {
            None
        } else {
            let goodtxconn = Connection::open(db_name)?;
            tune_sqlite_main(&goodtxconn)?;
            Some(goodtxconn)
        };
        let mut good_stage_shard_conns: Vec<Connection> = Vec::new();
        if use_good_sharding {
            for shard in 0..num_good_shards {
                let conn = Connection::open(tempgood_shard_path(shard))?;
                tune_sqlite_temp(&conn)?;
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS staged_good (
                        id INTEGER PRIMARY KEY,
                        cplx BLOB
                    )",
                    [],
                )?;
                good_stage_shard_conns.push(conn);
            }
        }
        let mut nall = 0;
        let mut frontier_direct_count = 0usize;
        let mut cc = 0;
        let ctot = ((lencplxs + *holder.chunksize - 1) / (*holder.chunksize)).max(1);
        let num_frontier_shards = if use_frontier_sharding { frontier_shards() } else { 1 };
        let use_bulk_label_dedup = bulk_label_dedup_enabled() || (use_frontier_sharding && !use_frontier_direct_dedup);
        let use_batch_merge_inserts = false;
        let use_frontier_canonical_representative =
            frontier_canonical_representative_enabled() && !use_batch_merge_inserts;
        let mut labelconn_opt = if use_frontier_sharding {
            None
        } else {
            let labelconn = Connection::open("templabels.db")?;
            tune_sqlite_temp(&labelconn)?;
            if use_frontier_canonical_representative {
                labelconn.execute(
                    if use_bulk_label_dedup {
                        "CREATE TABLE IF NOT EXISTS labeledcplxs (
                            id INTEGER PRIMARY KEY,
                            cplx BLOB
                        )"
                    } else {
                        "CREATE TABLE IF NOT EXISTS labeledcplxs (
                            cplx BLOB PRIMARY KEY,
                            id INTEGER
                        )"
                    },
                    [],
                )?;
            } else if use_bulk_label_dedup {
                labelconn.execute(
                    "CREATE TABLE IF NOT EXISTS labeledcplxs (
                        id INTEGER PRIMARY KEY,
                        label BLOB,
                        cplx BLOB
                    )",
                    [],
                )?;
            } else {
                labelconn.execute(
                    "CREATE TABLE IF NOT EXISTS labeledcplxs (
                        label BLOB PRIMARY KEY,
                        id INTEGER,
                        cplx BLOB
                    )",
                    [],
                )?;
            }
            Some(labelconn)
        };
        let mut label_shard_conns: Vec<Connection> = Vec::new();
        if use_frontier_sharding {
            for shard in 0..num_frontier_shards {
                let conn = Connection::open(templabel_shard_path(shard))?;
                tune_sqlite_temp(&conn)?;
                conn.execute(
                    if use_frontier_canonical_representative && use_frontier_direct_dedup {
                        "CREATE TABLE IF NOT EXISTS labeledcplxs (
                            id INTEGER PRIMARY KEY,
                            cplx BLOB UNIQUE
                        )"
                    } else if use_frontier_canonical_representative {
                        "CREATE TABLE IF NOT EXISTS labeledcplxs (
                            id INTEGER PRIMARY KEY,
                            cplx BLOB
                        )"
                    } else if use_frontier_direct_dedup {
                        "CREATE TABLE IF NOT EXISTS labeledcplxs (
                            id INTEGER PRIMARY KEY,
                            label BLOB UNIQUE,
                            cplx BLOB
                        )"
                    } else {
                        "CREATE TABLE IF NOT EXISTS labeledcplxs (
                            id INTEGER PRIMARY KEY,
                            label BLOB,
                            cplx BLOB
                        )"
                    },
                    [],
                )?;
                label_shard_conns.push(conn);
            }
        }
        // Keep batch-merge disabled while the sharded frontier pipeline stabilizes.
        let use_blob_encode_cache = blob_encode_cache_enabled();
        let use_stage_good_inserts = stage_good_inserts_enabled() && !use_batch_merge_inserts && !use_good_sharding;
        if use_stage_good_inserts {
            goodtxconn_opt.as_ref().expect("good main connection required").execute(
                "CREATE TEMP TABLE IF NOT EXISTS staged_good (
                    cplx BLOB,
                    id INTEGER
                )",
                [],
            )?;
        }
        if source_paths.len() <= 1 {
        for source_path in &source_paths {
            let cplxconn = Connection::open(source_path)?;
            tune_sqlite_temp(&cplxconn)?;
            let select_sql = format!(
                "SELECT id, cplx FROM cplxs{} WHERE id > ?1 ORDER BY id LIMIT ?2",
                i - 1
            );
            let mut stmt = cplxconn.prepare(&select_sql)?;
            let mut last_id = 0i64;

            'chunk_loop: loop {
                let gtx_opt = if use_good_sharding {
                    None
                } else {
                    Some(
                        goodtxconn_opt
                            .as_mut()
                            .expect("good main connection required")
                            .transaction()?,
                    )
                };
                let mut label_tx_opt = if use_frontier_sharding {
                    None
                } else {
                    Some(
                        labelconn_opt
                            .as_mut()
                            .expect("non-sharded mode requires label connection")
                            .transaction()?,
                    )
                };
                let mut label_txs_opt = if use_frontier_sharding {
                    let mut txs = Vec::with_capacity(num_frontier_shards);
                    for conn in label_shard_conns.iter_mut() {
                        txs.push(conn.transaction()?);
                    }
                    Some(txs)
                } else {
                    None
                };
                let mut stage_stmt1 = None;
                let mut label_mode = if use_frontier_sharding {
                    let mut stmts = Vec::with_capacity(num_frontier_shards);
                    for tx in label_txs_opt
                        .as_mut()
                        .expect("sharded mode requires label transactions")
                        .iter_mut()
                    {
                        stmts.push(tx.prepare(
                            if use_frontier_canonical_representative && use_frontier_direct_dedup {
                                "INSERT OR IGNORE INTO labeledcplxs (id, cplx) VALUES (?, ?)"
                            } else if use_frontier_canonical_representative {
                                "INSERT INTO labeledcplxs (id, cplx) VALUES (?, ?)"
                            } else if use_frontier_direct_dedup {
                                "INSERT OR IGNORE INTO labeledcplxs (id, label, cplx) VALUES (?, ?, ?)"
                            } else {
                                "INSERT INTO labeledcplxs (id, label, cplx) VALUES (?, ?, ?)"
                            }
                        )?);
                    }
                    LabelInsertMode::Sharded(stmts)
                } else {
                    let tx = label_tx_opt
                        .as_mut()
                        .expect("non-sharded mode requires label transaction");
                    if use_batch_merge_inserts {
                        stage_stmt1 = Some(tx.prepare(
                            if use_frontier_canonical_representative {
                                "INSERT INTO chunk_labeled (cplx, id) VALUES (?, ?)"
                            } else {
                                "INSERT INTO chunk_labeled (label, id, cplx) VALUES (?, ?, ?)"
                            }
                        )?);
                    }
                    if use_frontier_canonical_representative && use_bulk_label_dedup {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT INTO labeledcplxs (id, cplx) VALUES (?, ?)")?
                        )
                    } else if use_frontier_canonical_representative {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT OR IGNORE INTO labeledcplxs (cplx, id) VALUES (?, ?)")?
                        )
                    } else if use_bulk_label_dedup {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT INTO labeledcplxs (id, label, cplx) VALUES (?, ?, ?)")?
                        )
                    } else {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT OR IGNORE INTO labeledcplxs (label, id, cplx) VALUES (?, ?, ?)")?
                        )
                    }
                };
                let mut stmt2_opt = if use_good_sharding {
                    None
                } else {
                    Some(
                        gtx_opt
                            .as_ref()
                            .expect("good main transaction required")
                            .prepare("INSERT OR IGNORE INTO goodcplxs (cplx, id) VALUES (?, ?)")?
                    )
                };
                let mut stage_stmt2 = if use_batch_merge_inserts {
                    Some(
                        gtx_opt
                            .as_ref()
                            .expect("good main transaction required")
                            .prepare("INSERT INTO chunk_good (cplx, id) VALUES (?, ?)")?
                    )
                } else {
                    None
                };
                let mut stage_good_stmt = if use_stage_good_inserts {
                    gtx_opt
                        .as_ref()
                        .expect("good main transaction required")
                        .execute("DELETE FROM staged_good", [])?;
                    Some(
                        gtx_opt
                            .as_ref()
                            .expect("good main transaction required")
                            .prepare("INSERT INTO staged_good (cplx, id) VALUES (?, ?)")?
                    )
                } else {
                    None
                };
                let mut good_stage_shard_txs_opt = if use_good_sharding {
                    let mut txs = Vec::with_capacity(num_good_shards);
                    for conn in good_stage_shard_conns.iter_mut() {
                        txs.push(conn.transaction()?);
                    }
                    Some(txs)
                } else {
                    None
                };
                let mut good_stage_shard_stmts_opt = if use_good_sharding {
                    let mut stmts = Vec::with_capacity(num_good_shards);
                    for tx in good_stage_shard_txs_opt
                        .as_mut()
                        .expect("good sharding requires shard transactions")
                        .iter_mut()
                    {
                        stmts.push(tx.prepare("INSERT INTO staged_good (id, cplx) VALUES (?, ?)")?);
                    }
                    Some(stmts)
                } else {
                    None
                };
                let mut processed_in_tx = 0usize;
                let mut no_more_rows = false;

                loop {
                    let read_start = Instant::now();
                    let rows_iter = stmt.query_map(params![last_id, *holder.chunksize as i64], |row| {
                        let id: i64 = row.get(0)?;
                        let cplx_blob: Vec<u8> = row.get(1)?;
                        Ok((id, decode_temp_bits(&cplx_blob, &holder)))
                    })?;

                    let chunk_rows: Result<Vec<(i64, B)>, rusqlite::Error> = rows_iter.collect();
                    let chunk_rows: Vec<(i64, B)> = chunk_rows?;
                    let read_ms = read_start.elapsed().as_millis();
                    if chunk_rows.is_empty() {
                        no_more_rows = true;
                        break;
                    }
                    last_id = chunk_rows.last().expect("Chunk is not empty.").0;
                    let chunk: Vec<B> = chunk_rows.into_iter().map(|(_, c)| c).collect();

                    cc += 1;
                    print!("{: <85}\r", "");
                    print!("{}: Chunk {}/{} [read={}ms]. ", Local::now().format("%Y-%m-%d %H:%M:%S"), cc, ctot, read_ms);
                    io::stdout().flush().unwrap();
                    let process_start = Instant::now();
                    let stats = process_chunk(
                        &mut label_mode,
                        stmt2_opt.as_mut(),
                        stage_stmt1.as_mut(),
                        stage_stmt2.as_mut(),
                        stage_good_stmt.as_mut(),
                        good_stage_shard_stmts_opt.as_mut(),
                        &holder,
                        &chunk,
                        &mut nall,
                        &mut ngood,
                        dbprefix,
                        use_batch_merge_inserts,
                        use_blob_encode_cache,
                        use_stage_good_inserts,
                        use_good_sharding,
                        use_bulk_label_dedup,
                        use_frontier_sharding,
                        use_frontier_direct_dedup,
                        use_frontier_canonical_representative,
                        num_frontier_shards,
                        num_good_shards,
                    )?;
                    if use_frontier_direct_dedup {
                        frontier_direct_count += stats.inserted_labeled;
                    }
                    let process_ms = process_start.elapsed().as_millis();
                    if let Some(log) = profile_log.as_mut() {
                        let _ = writeln!(
                            log,
                            "{} level={} chunk={}/{} read_ms={} analyze_ms={} reduce_ms={} proc_ms={} generated={} labeled_ins={} good_ins={}",
                            Local::now().format("%Y-%m-%d %H:%M:%S"),
                            i,
                            cc,
                            ctot,
                            read_ms,
                            stats.analyze_ms,
                            stats.reduce_ms,
                            process_ms,
                            stats.generated,
                            stats.inserted_labeled,
                            stats.inserted_good
                        );
                    }

                    processed_in_tx += 1;
                    if processed_in_tx >= COMMIT_EVERY_CHUNKS {
                        print!("Proc({}ms). ", process_ms);
                        io::stdout().flush().unwrap();
                        break;
                    }
                    print!("Proc({}ms). ", process_ms);
                    io::stdout().flush().unwrap();
                }

                drop(stmt2_opt);
                drop(label_mode);
                drop(stage_stmt2);
                drop(stage_stmt1);
                drop(stage_good_stmt);
                drop(good_stage_shard_stmts_opt);
                if use_stage_good_inserts {
                    gtx_opt
                        .as_ref()
                        .expect("good main transaction required")
                        .execute(
                        "INSERT OR IGNORE INTO goodcplxs (cplx, id)
                         SELECT cplx, id FROM staged_good",
                        [],
                    )?;
                    gtx_opt
                        .as_ref()
                        .expect("good main transaction required")
                        .execute("DELETE FROM staged_good", [])?;
                }
                let commit_start = Instant::now();
                if let Some(good_stage_txs) = good_stage_shard_txs_opt {
                    for tx in good_stage_txs {
                        tx.commit()?;
                    }
                }
                if let Some(tx) = label_tx_opt {
                    tx.commit()?;
                }
                if let Some(label_txs) = label_txs_opt {
                    for tx in label_txs {
                        tx.commit()?;
                    }
                }
                if let Some(gtx) = gtx_opt {
                    gtx.commit()?;
                }
                let commit_ms = commit_start.elapsed().as_millis();
                if let Some(log) = profile_log.as_mut() {
                    let _ = writeln!(
                        log,
                        "{} level={} commit_ms={} chunks_in_tx={}",
                        Local::now().format("%Y-%m-%d %H:%M:%S"),
                        i,
                        commit_ms,
                        processed_in_tx
                    );
                }
                print!("Commit({}ms).\r", commit_ms);
                io::stdout().flush().unwrap();
                if no_more_rows {
                    break 'chunk_loop;
                }
            }
            drop(stmt);
            drop(cplxconn);
        }
        } else {
            let mut cursors: Vec<SourceCursor<B>> = Vec::new();
            for source_path in &source_paths {
                let conn = Connection::open(source_path)?;
                tune_sqlite_temp(&conn)?;
                cursors.push(SourceCursor {
                    conn,
                    path: source_path.clone(),
                    last_id: 0,
                    buf: Vec::new(),
                    pos: 0,
                    exhausted: false,
                });
            }

            'chunk_loop: loop {
                let gtx_opt = if use_good_sharding {
                    None
                } else {
                    Some(
                        goodtxconn_opt
                            .as_mut()
                            .expect("good main connection required")
                            .transaction()?,
                    )
                };
                let mut label_tx_opt = if use_frontier_sharding {
                    None
                } else {
                    Some(
                        labelconn_opt
                            .as_mut()
                            .expect("non-sharded mode requires label connection")
                            .transaction()?,
                    )
                };
                let mut label_txs_opt = if use_frontier_sharding {
                    let mut txs = Vec::with_capacity(num_frontier_shards);
                    for conn in label_shard_conns.iter_mut() {
                        txs.push(conn.transaction()?);
                    }
                    Some(txs)
                } else {
                    None
                };
                let mut stage_stmt1 = None;
                let mut label_mode = if use_frontier_sharding {
                    let mut stmts = Vec::with_capacity(num_frontier_shards);
                    for tx in label_txs_opt
                        .as_mut()
                        .expect("sharded mode requires label transactions")
                        .iter_mut()
                    {
                        stmts.push(tx.prepare(
                            if use_frontier_canonical_representative && use_frontier_direct_dedup {
                                "INSERT OR IGNORE INTO labeledcplxs (id, cplx) VALUES (?, ?)"
                            } else if use_frontier_canonical_representative {
                                "INSERT INTO labeledcplxs (id, cplx) VALUES (?, ?)"
                            } else if use_frontier_direct_dedup {
                                "INSERT OR IGNORE INTO labeledcplxs (id, label, cplx) VALUES (?, ?, ?)"
                            } else {
                                "INSERT INTO labeledcplxs (id, label, cplx) VALUES (?, ?, ?)"
                            }
                        )?);
                    }
                    LabelInsertMode::Sharded(stmts)
                } else {
                    let tx = label_tx_opt
                        .as_mut()
                        .expect("non-sharded mode requires label transaction");
                    if use_batch_merge_inserts {
                        stage_stmt1 = Some(tx.prepare(
                            if use_frontier_canonical_representative {
                                "INSERT INTO chunk_labeled (cplx, id) VALUES (?, ?)"
                            } else {
                                "INSERT INTO chunk_labeled (label, id, cplx) VALUES (?, ?, ?)"
                            }
                        )?);
                    }
                    if use_frontier_canonical_representative && use_bulk_label_dedup {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT INTO labeledcplxs (id, cplx) VALUES (?, ?)")?
                        )
                    } else if use_frontier_canonical_representative {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT OR IGNORE INTO labeledcplxs (cplx, id) VALUES (?, ?)")?
                        )
                    } else if use_bulk_label_dedup {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT INTO labeledcplxs (id, label, cplx) VALUES (?, ?, ?)")?
                        )
                    } else {
                        LabelInsertMode::Single(
                            tx.prepare("INSERT OR IGNORE INTO labeledcplxs (label, id, cplx) VALUES (?, ?, ?)")?
                        )
                    }
                };
                let mut stmt2_opt = if use_good_sharding {
                    None
                } else {
                    Some(
                        gtx_opt
                            .as_ref()
                            .expect("good main transaction required")
                            .prepare("INSERT OR IGNORE INTO goodcplxs (cplx, id) VALUES (?, ?)")?
                    )
                };
                let mut stage_stmt2 = if use_batch_merge_inserts {
                    Some(
                        gtx_opt
                            .as_ref()
                            .expect("good main transaction required")
                            .prepare("INSERT INTO chunk_good (cplx, id) VALUES (?, ?)")?
                    )
                } else {
                    None
                };
                let mut stage_good_stmt = if use_stage_good_inserts {
                    gtx_opt
                        .as_ref()
                        .expect("good main transaction required")
                        .execute("DELETE FROM staged_good", [])?;
                    Some(
                        gtx_opt
                            .as_ref()
                            .expect("good main transaction required")
                            .prepare("INSERT INTO staged_good (cplx, id) VALUES (?, ?)")?
                    )
                } else {
                    None
                };
                let mut good_stage_shard_txs_opt = if use_good_sharding {
                    let mut txs = Vec::with_capacity(num_good_shards);
                    for conn in good_stage_shard_conns.iter_mut() {
                        txs.push(conn.transaction()?);
                    }
                    Some(txs)
                } else {
                    None
                };
                let mut good_stage_shard_stmts_opt = if use_good_sharding {
                    let mut stmts = Vec::with_capacity(num_good_shards);
                    for tx in good_stage_shard_txs_opt
                        .as_mut()
                        .expect("good sharding requires shard transactions")
                        .iter_mut()
                    {
                        stmts.push(tx.prepare("INSERT INTO staged_good (id, cplx) VALUES (?, ?)")?);
                    }
                    Some(stmts)
                } else {
                    None
                };
                let mut processed_in_tx = 0usize;
                let mut no_more_rows = false;

                loop {
                    let read_start = Instant::now();
                    let mut chunk: Vec<B> = Vec::with_capacity(*holder.chunksize);
                    while chunk.len() < *holder.chunksize {
                        let mut best_idx: Option<usize> = None;
                        let mut best_id: i64 = i64::MAX;

                        for idx in 0..cursors.len() {
                            if cursors[idx].exhausted {
                                continue;
                            }
                            if cursors[idx].pos >= cursors[idx].buf.len() {
                                let rows = fetch_source_batch::<B>(
                                    &cursors[idx].conn,
                                    i - 1,
                                    cursors[idx].last_id,
                                    *holder.chunksize,
                                    &holder,
                                )?;
                                if rows.is_empty() {
                                    cursors[idx].exhausted = true;
                                    continue;
                                }
                                cursors[idx].last_id = rows.last().expect("rows not empty").0;
                                cursors[idx].buf = rows;
                                cursors[idx].pos = 0;
                            }
                            let id = cursors[idx].buf[cursors[idx].pos].0;
                            if id < best_id {
                                best_id = id;
                                best_idx = Some(idx);
                            }
                        }

                        match best_idx {
                            Some(idx) => {
                                let (_, cplx) = cursors[idx].buf[cursors[idx].pos];
                                cursors[idx].pos += 1;
                                if cursors[idx].pos >= cursors[idx].buf.len() {
                                    cursors[idx].buf.clear();
                                    cursors[idx].pos = 0;
                                }
                                chunk.push(cplx);
                            }
                            None => break,
                        }
                    }

                    let read_ms = read_start.elapsed().as_millis();
                    if chunk.is_empty() {
                        no_more_rows = true;
                        break;
                    }

                    cc += 1;
                    print!("{: <85}\r", "");
                    print!("{}: Chunk {}/{} [read={}ms]. ", Local::now().format("%Y-%m-%d %H:%M:%S"), cc, ctot, read_ms);
                    io::stdout().flush().unwrap();
                    let process_start = Instant::now();
                    let stats = process_chunk(
                        &mut label_mode,
                        stmt2_opt.as_mut(),
                        stage_stmt1.as_mut(),
                        stage_stmt2.as_mut(),
                        stage_good_stmt.as_mut(),
                        good_stage_shard_stmts_opt.as_mut(),
                        &holder,
                        &chunk,
                        &mut nall,
                        &mut ngood,
                        dbprefix,
                        use_batch_merge_inserts,
                        use_blob_encode_cache,
                        use_stage_good_inserts,
                        use_good_sharding,
                        use_bulk_label_dedup,
                        use_frontier_sharding,
                        use_frontier_direct_dedup,
                        use_frontier_canonical_representative,
                        num_frontier_shards,
                        num_good_shards,
                    )?;
                    if use_frontier_direct_dedup {
                        frontier_direct_count += stats.inserted_labeled;
                    }
                    let process_ms = process_start.elapsed().as_millis();
                    if let Some(log) = profile_log.as_mut() {
                        let _ = writeln!(
                            log,
                            "{} level={} chunk={}/{} read_ms={} analyze_ms={} reduce_ms={} proc_ms={} generated={} labeled_ins={} good_ins={}",
                            Local::now().format("%Y-%m-%d %H:%M:%S"),
                            i,
                            cc,
                            ctot,
                            read_ms,
                            stats.analyze_ms,
                            stats.reduce_ms,
                            process_ms,
                            stats.generated,
                            stats.inserted_labeled,
                            stats.inserted_good
                        );
                    }

                    processed_in_tx += 1;
                    print!("Proc({}ms). ", process_ms);
                    io::stdout().flush().unwrap();
                    if processed_in_tx >= COMMIT_EVERY_CHUNKS {
                        break;
                    }
                }

                drop(stmt2_opt);
                drop(label_mode);
                drop(stage_stmt2);
                drop(stage_stmt1);
                drop(stage_good_stmt);
                drop(good_stage_shard_stmts_opt);
                if use_stage_good_inserts {
                    gtx_opt
                        .as_ref()
                        .expect("good main transaction required")
                        .execute(
                        "INSERT OR IGNORE INTO goodcplxs (cplx, id)
                         SELECT cplx, id FROM staged_good",
                        [],
                    )?;
                    gtx_opt
                        .as_ref()
                        .expect("good main transaction required")
                        .execute("DELETE FROM staged_good", [])?;
                }
                let commit_start = Instant::now();
                if let Some(good_stage_txs) = good_stage_shard_txs_opt {
                    for tx in good_stage_txs {
                        tx.commit()?;
                    }
                }
                if let Some(tx) = label_tx_opt {
                    tx.commit()?;
                }
                if let Some(label_txs) = label_txs_opt {
                    for tx in label_txs {
                        tx.commit()?;
                    }
                }
                if let Some(gtx) = gtx_opt {
                    gtx.commit()?;
                }
                let commit_ms = commit_start.elapsed().as_millis();
                if let Some(log) = profile_log.as_mut() {
                    let _ = writeln!(
                        log,
                        "{} level={} commit_ms={} chunks_in_tx={}",
                        Local::now().format("%Y-%m-%d %H:%M:%S"),
                        i,
                        commit_ms,
                        processed_in_tx
                    );
                }
                print!("Commit({}ms).\r", commit_ms);
                io::stdout().flush().unwrap();
                if no_more_rows {
                    break 'chunk_loop;
                }
            }

            for cursor in cursors {
                let _ = cursor.path;
                drop(cursor);
            }
        }

        print!("{: <85}\r", "");
        print!("{}: Dropping table ...         \r", Local::now().format("%Y-%m-%d %H:%M:%S"));
        io::stdout().flush().unwrap();

        let finalize_total_start = Instant::now();
        let source_drop_ms: u128;
        let mut frontier_finalize = FinalizePhaseTimings::default();
        let mut good_finalize = FinalizePhaseTimings::default();
        let mut good_count_ms: u128 = 0;
        let mut main_checkpoint_ms: u128 = 0;

        let source_temp_db_size: u64 = source_paths.iter().map(|p| file_size(p)).sum();
        let source_temp_wal_size: u64 = source_paths
            .iter()
            .map(|p| file_size(&format!("{p}-wal")))
            .sum();

        let source_drop_start = Instant::now();
        for source_path in &source_paths {
            remove_sqlite_artifacts(source_path);
        }
        drop(good_stage_shard_conns);
        source_drop_ms = source_drop_start.elapsed().as_millis();
        let staged_good_db_size_before_finalize: u64 = if use_good_sharding {
            (0..num_good_shards).map(|shard| file_size(&tempgood_shard_path(shard))).sum()
        } else {
            0
        };
        let staged_good_wal_size_before_finalize: u64 = if use_good_sharding {
            (0..num_good_shards)
                .map(|shard| file_size(&format!("{}-wal", tempgood_shard_path(shard))))
                .sum()
        } else {
            0
        };
        let good_db_size_before_finalize: u64 = if use_good_sharding {
            (0..num_good_shards).map(|shard| file_size(&goodcplx_shard_path(&good_db_stem, shard))).sum()
        } else {
            0
        };
        let good_wal_size_before_finalize: u64 = if use_good_sharding {
            (0..num_good_shards)
                .map(|shard| file_size(&format!("{}-wal", goodcplx_shard_path(&good_db_stem, shard))))
                .sum()
        } else {
            0
        };

        let materialize_next_frontier = i < level_stop;
        let mut next_paths: Vec<String> = Vec::new();
        let next_count: usize;
        let mut label_paths: Vec<String> = Vec::new();
        let mut label_temp_db_size: u64 = 0;
        let mut label_temp_wal_size: u64 = 0;
        if materialize_next_frontier {
            if use_frontier_direct_dedup {
                print!("{}: Finalizing frontier ...    \r", Local::now().format("%Y-%m-%d %H:%M:%S"));
            } else {
                print!("{}: Copying table ...          \r", Local::now().format("%Y-%m-%d %H:%M:%S"));
            }
            io::stdout().flush().unwrap();
        }
        if use_frontier_sharding && materialize_next_frontier {
            let frontier_total_start = Instant::now();
            drop(label_shard_conns);
            let mut shard_output_paths: Vec<String> = Vec::new();
            let mut shard_total_count: usize = 0;
            for shard in 0..num_frontier_shards {
                let labelpath = templabel_shard_path(shard);
                label_paths.push(labelpath.clone());
                label_temp_db_size += file_size(&labelpath);
                label_temp_wal_size += file_size(&format!("{labelpath}-wal"));
                if use_frontier_direct_dedup {
                    let out_path = tempcplx_shard_path(i, shard);
                    let setup_start = Instant::now();
                    let cplxconn = Connection::open(&labelpath)?;
                    cplxconn.execute(
                        &format!("ALTER TABLE labeledcplxs RENAME TO cplxs{}", i),
                        [],
                    )?;
                    frontier_finalize.setup_ms += setup_start.elapsed().as_millis();
                    let rename_start = Instant::now();
                    drop(cplxconn);
                    rename_sqlite_artifacts(&labelpath, &out_path)?;
                    frontier_finalize.rename_ms += rename_start.elapsed().as_millis();
                    shard_output_paths.push(out_path);
                } else {
                    let out_path = tempcplx_shard_path(i, shard);
                    let setup_start = Instant::now();
                    let cplxconn = Connection::open(&out_path)?;
                    tune_sqlite_finalize_output(&cplxconn)?;
                    cplxconn.execute(
                        &format!(
                            "CREATE TABLE cplxs{} (
                                        id INTEGER PRIMARY KEY,
                                        cplx BLOB
                            )",
                            i
                        ),
                        [],
                    )?;
                    frontier_finalize.setup_ms += setup_start.elapsed().as_millis();
                    if finalize_checkpoint_labels_enabled() {
                        let checkpoint_start = Instant::now();
                        checkpoint_temp_db(&labelpath)?;
                        frontier_finalize.checkpoint_ms += checkpoint_start.elapsed().as_millis();
                    }
                    let attach_start = Instant::now();
                    cplxconn.execute("ATTACH DATABASE ? AS templabels", params![labelpath])?;
                    frontier_finalize.attach_ms += attach_start.elapsed().as_millis();
                    let insert_start = Instant::now();
                    let shard_count = if use_frontier_canonical_representative {
                        cplxconn.execute(
                            &format!(
                                "INSERT INTO cplxs{} (id,cplx)
                                 SELECT MIN(id) AS id, cplx
                                 FROM templabels.labeledcplxs
                                 GROUP BY cplx
                                 ORDER BY id",
                                i
                            ),
                            [],
                        )?
                    } else {
                        let dedup_start = Instant::now();
                        cplxconn.execute(
                            "CREATE TEMP TABLE dedup_ids AS
                             SELECT MIN(id) AS id
                             FROM templabels.labeledcplxs
                             GROUP BY label",
                            [],
                        )?;
                        frontier_finalize.dedup_ms += dedup_start.elapsed().as_millis();
                        cplxconn.execute(
                            &format!(
                                "INSERT INTO cplxs{} (id,cplx)
                                 SELECT l.id, l.cplx
                                 FROM templabels.labeledcplxs AS l
                                 JOIN dedup_ids AS d ON l.id = d.id
                                 ORDER BY l.id",
                                i
                            ),
                            [],
                        )?
                    };
                    frontier_finalize.insert_ms += insert_start.elapsed().as_millis();
                    let cleanup_start = Instant::now();
                    if !use_frontier_canonical_representative {
                        cplxconn.execute("DROP TABLE dedup_ids", [])?;
                    }
                    cplxconn.execute("DETACH DATABASE templabels", [])?;
                    drop(cplxconn);
                    if shard_count > 0 {
                        shard_total_count += shard_count;
                        shard_output_paths.push(out_path);
                    } else {
                        remove_sqlite_artifacts(&out_path);
                    }
                    remove_sqlite_artifacts(&labelpath);
                    frontier_finalize.cleanup_ms += cleanup_start.elapsed().as_millis();
                }
            }
            frontier_finalize.total_ms = frontier_total_start.elapsed().as_millis();
            next_count = if use_frontier_direct_dedup {
                frontier_direct_count
            } else {
                shard_total_count
            };
            next_paths = shard_output_paths;
        } else if !use_frontier_sharding && materialize_next_frontier {
            let frontier_total_start = Instant::now();
            let setup_start = Instant::now();
            let cplxconn = Connection::open(tempcplx_path(i))?;
            tune_sqlite_finalize_output(&cplxconn)?;
            cplxconn.execute(
                &format!(
                    "CREATE TABLE cplxs{} (
                                id INTEGER PRIMARY KEY,
                                cplx BLOB
            )",
                    i
                ),
                [],
            )?;
            frontier_finalize.setup_ms += setup_start.elapsed().as_millis();

            let labelpath = "templabels.db";
            label_paths.push(labelpath.to_string());
            if finalize_checkpoint_labels_enabled() {
                let checkpoint_start = Instant::now();
                checkpoint_temp_db(labelpath)?;
                frontier_finalize.checkpoint_ms += checkpoint_start.elapsed().as_millis();
            }
            let attach_start = Instant::now();
            cplxconn.execute("ATTACH DATABASE ? AS templabels", params![labelpath])?;
            frontier_finalize.attach_ms += attach_start.elapsed().as_millis();

            if use_bulk_label_dedup {
                let insert_start = Instant::now();
                next_count = if use_frontier_canonical_representative {
                    cplxconn.execute(
                        &format!(
                            "INSERT INTO cplxs{} (id,cplx)
                             SELECT MIN(id) AS id, cplx
                             FROM templabels.labeledcplxs
                             GROUP BY cplx
                             ORDER BY id",
                            i
                        ),
                        [],
                    )?
                } else {
                    let dedup_start = Instant::now();
                    cplxconn.execute(
                        "CREATE TEMP TABLE dedup_ids AS
                         SELECT MIN(id) AS id
                         FROM templabels.labeledcplxs
                         GROUP BY label",
                        [],
                    )?;
                    frontier_finalize.dedup_ms += dedup_start.elapsed().as_millis();
                    cplxconn.execute(
                        &format!(
                            "INSERT INTO cplxs{} (id,cplx)
                             SELECT l.id, l.cplx
                             FROM templabels.labeledcplxs AS l
                             JOIN dedup_ids AS d ON l.id = d.id
                             ORDER BY l.id",
                            i
                        ),
                        [],
                    )?
                };
                frontier_finalize.insert_ms += insert_start.elapsed().as_millis();
                let cleanup_start = Instant::now();
                if !use_frontier_canonical_representative {
                    cplxconn.execute("DROP TABLE dedup_ids", [])?;
                }
                frontier_finalize.cleanup_ms += cleanup_start.elapsed().as_millis();
            } else {
                let insert_start = Instant::now();
                next_count = cplxconn.execute(
                    &format!(
                        "INSERT INTO cplxs{} (id,cplx)
                         SELECT id, cplx
                         FROM templabels.labeledcplxs",
                        i
                    ),
                    [],
                )?;
                frontier_finalize.insert_ms += insert_start.elapsed().as_millis();
            }

            let cleanup_start = Instant::now();
            cplxconn.execute("DETACH DATABASE templabels", [])?;
            next_paths.push(tempcplx_path(i));
            print!("{}: Dropping labeledcplxs ...\r", Local::now().format("%Y-%m-%d %H:%M:%S"));
            io::stdout().flush().unwrap();
            label_temp_db_size = file_size(labelpath);
            label_temp_wal_size = file_size(&format!("{labelpath}-wal"));
            drop(labelconn_opt.take());
            remove_sqlite_artifacts("templabels.db");
            drop(cplxconn);
            frontier_finalize.cleanup_ms += cleanup_start.elapsed().as_millis();
            frontier_finalize.total_ms = frontier_total_start.elapsed().as_millis();
        } else if use_frontier_sharding {
            let frontier_total_start = Instant::now();
            drop(label_shard_conns);
            print!("{}: Dropping labeledcplxs ...\r", Local::now().format("%Y-%m-%d %H:%M:%S"));
            io::stdout().flush().unwrap();
            let cleanup_start = Instant::now();
            for shard in 0..num_frontier_shards {
                let labelpath = templabel_shard_path(shard);
                label_paths.push(labelpath.clone());
                label_temp_db_size += file_size(&labelpath);
                label_temp_wal_size += file_size(&format!("{labelpath}-wal"));
                remove_sqlite_artifacts(&labelpath);
            }
            frontier_finalize.cleanup_ms += cleanup_start.elapsed().as_millis();
            frontier_finalize.total_ms = frontier_total_start.elapsed().as_millis();
            next_count = 0;
        } else {
            let frontier_total_start = Instant::now();
            let labelpath = "templabels.db";
            label_paths.push(labelpath.to_string());
            print!("{}: Dropping labeledcplxs ...\r", Local::now().format("%Y-%m-%d %H:%M:%S"));
            io::stdout().flush().unwrap();
            let cleanup_start = Instant::now();
            label_temp_db_size = file_size(labelpath);
            label_temp_wal_size = file_size(&format!("{labelpath}-wal"));
            drop(labelconn_opt.take());
            remove_sqlite_artifacts(labelpath);
            frontier_finalize.cleanup_ms += cleanup_start.elapsed().as_millis();
            frontier_finalize.total_ms = frontier_total_start.elapsed().as_millis();
            next_count = 0;
        }

        if use_frontier_sharding && materialize_next_frontier {
            print!("{}: Dropping labeledcplxs ...\r", Local::now().format("%Y-%m-%d %H:%M:%S"));
            io::stdout().flush().unwrap();
        }

        lencplxs = if materialize_next_frontier { next_count } else { nall };
        if use_good_sharding {
            let (inserted_new_good, timings) = finalize_good_shards(num_good_shards, &good_db_stem)?;
            good_finalize = timings;
            lengoodcplxs += inserted_new_good;
        } else {
            let good_count_start = Instant::now();
            lengoodcplxs = conn.query_row("SELECT COUNT(*) FROM goodcplxs", [], |row| row.get(0),).expect("Failed.");
            good_count_ms = good_count_start.elapsed().as_millis();
        }
        if let Some(mode) = checkpoint_mode() {
            let level_idx = i.saturating_sub(imin) + 1;
            if level_idx % checkpoint_every_levels() == 0 {
                let checkpoint_start = Instant::now();
                let _ = conn.execute(&format!("PRAGMA wal_checkpoint({mode})"), []);
                main_checkpoint_ms = checkpoint_start.elapsed().as_millis();
            }
        }
        let finalize_total_ms = finalize_total_start.elapsed().as_millis();
        if let Some(log) = profile_log.as_mut() {
            let main_wal = format!("{}-wal", db_name);
            let next_temp_db_size: u64 = next_paths.iter().map(|p| file_size(p)).sum();
            let next_temp_wal_size: u64 = next_paths
                .iter()
                .map(|p| file_size(&format!("{p}-wal")))
                .sum();
            let main_db_size = file_size(db_name);
            let main_wal_size = file_size(&main_wal);
            let good_db_size_after_finalize: u64 = if use_good_sharding {
                (0..num_good_shards).map(|shard| file_size(&goodcplx_shard_path(&good_db_stem, shard))).sum()
            } else {
                0
            };
            let good_wal_size_after_finalize: u64 = if use_good_sharding {
                (0..num_good_shards)
                    .map(|shard| file_size(&format!("{}-wal", goodcplx_shard_path(&good_db_stem, shard))))
                    .sum()
            } else {
                0
            };
            let estimated_live_total_before_finalize = main_db_size
                + main_wal_size
                + source_temp_db_size
                + source_temp_wal_size
                + label_temp_db_size
                + label_temp_wal_size
                + good_db_size_before_finalize
                + good_wal_size_before_finalize
                + staged_good_db_size_before_finalize
                + staged_good_wal_size_before_finalize;
            let estimated_live_total_after_finalize = main_db_size
                + main_wal_size
                + next_temp_db_size
                + next_temp_wal_size
                + good_db_size_after_finalize
                + good_wal_size_after_finalize;
            let estimated_live_total_peak = std::cmp::max(
                estimated_live_total_before_finalize,
                estimated_live_total_after_finalize,
            );
            let _ = writeln!(
                log,
                "{} level={} finalize_total_ms={} source_drop_ms={} frontier_total_ms={} frontier_setup_ms={} frontier_checkpoint_ms={} frontier_attach_ms={} frontier_dedup_ms={} frontier_insert_ms={} frontier_rename_ms={} frontier_cleanup_ms={} good_total_ms={} good_setup_ms={} good_checkpoint_ms={} good_attach_ms={} good_dedup_ms={} good_insert_ms={} good_rename_ms={} good_cleanup_ms={} good_count_ms={} post_checkpoint_ms={}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                i,
                finalize_total_ms,
                source_drop_ms,
                frontier_finalize.total_ms,
                frontier_finalize.setup_ms,
                frontier_finalize.checkpoint_ms,
                frontier_finalize.attach_ms,
                frontier_finalize.dedup_ms,
                frontier_finalize.insert_ms,
                frontier_finalize.rename_ms,
                frontier_finalize.cleanup_ms,
                good_finalize.total_ms,
                good_finalize.setup_ms,
                good_finalize.checkpoint_ms,
                good_finalize.attach_ms,
                good_finalize.dedup_ms,
                good_finalize.insert_ms,
                good_finalize.rename_ms,
                good_finalize.cleanup_ms,
                good_count_ms,
                main_checkpoint_ms,
            );
            let _ = writeln!(
                log,
                "{} level={} size_main_db={} size_main_wal={} size_temp_db={} size_temp_wal={}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                i,
                main_db_size,
                main_wal_size,
                next_temp_db_size,
                next_temp_wal_size
            );
            let _ = writeln!(
                log,
                "{} level={} source_temp_db={} source_temp_wal={} label_temp_db={} label_temp_wal={} staged_good_db={} staged_good_wal={} good_db_before_finalize={} good_wal_before_finalize={} next_temp_db={} next_temp_wal={} good_db_after_finalize={} good_wal_after_finalize={} est_live_before_finalize={} est_live_after_finalize={} est_live_peak={}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                i,
                source_temp_db_size,
                source_temp_wal_size,
                label_temp_db_size,
                label_temp_wal_size,
                staged_good_db_size_before_finalize,
                staged_good_wal_size_before_finalize,
                good_db_size_before_finalize,
                good_wal_size_before_finalize,
                next_temp_db_size,
                next_temp_wal_size,
                good_db_size_after_finalize,
                good_wal_size_after_finalize,
                estimated_live_total_before_finalize,
                estimated_live_total_after_finalize,
                estimated_live_total_peak
            );
            let (prune_global_coverage, prune_edge_neighborhood) = functions::read_extend_prune_counters();
            let _ = writeln!(
                log,
                "{} level={} prune_global_coverage={} prune_edge_neighborhood={} global_coverage_prune={} edge_neighborhood_prune={}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                i,
                prune_global_coverage,
                prune_edge_neighborhood,
                functions::global_coverage_prune_enabled(),
                functions::edge_neighborhood_prune_enabled()
            );
            let (canon_calls, canon_root_candidates, canon_tau_trials) = functions::read_canon_counters();
            let _ = writeln!(
                log,
                "{} level={} canon_calls={} canon_root_candidates={} canon_tau_trials={}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                i,
                canon_calls,
                canon_root_candidates,
                canon_tau_trials
            );
        }

        print!("{: <85}\r", "");
        println!("{}: # squares: {}; # cplxs: {}; # goodcplxs: {}", Local::now().format("%Y-%m-%d %H:%M:%S"), i, lencplxs, lengoodcplxs);
        if lencplxs == 0 {
            print!("Cleaning up ...\r");
            io::stdout().flush().unwrap();
            for next_path in &next_paths {
                remove_sqlite_artifacts(next_path);
            }
            let _ = conn.execute("VACUUM",[]);
            break;
        }
    }
    if use_good_sharding && good_final_merge_enabled() {
        print!("Merging good shards ...\r");
        io::stdout().flush().unwrap();
        merge_good_shards_into_main(&conn, num_good_shards, &good_db_stem)?;
        for shard in 0..num_good_shards {
            remove_sqlite_artifacts(&goodcplx_shard_path(&good_db_stem, shard));
        }
    } else if use_good_sharding {
        println!(
            "Final good output left sharded as {}_good_part_XX.db ({} shards).",
            good_db_stem,
            num_good_shards
        );
    }
    println!("Total number of goodcplxs: {}",lengoodcplxs);
    if let Some(log) = profile_log.as_mut() {
        let _ = writeln!(
            log,
            "{} run_end",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
        );
    }
    Ok(())
}

#[pymodule]
fn rust_code(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(extendonce, m)?)?;
    // m.add_function(wrap_pyfunction!(disconnected_withbdry_extendonce, m)?)?;
    // m.add_function(wrap_pyfunction!(withbdry_extendonce, m)?)?;
    // m.add_function(wrap_pyfunction!(disconnected_extendonce, m)?)?;
    // m.add_function(wrap_pyfunction!(cubicalcanlabel, m)?)?;
    // m.add_function(wrap_pyfunction!(testedges, m)?)?;
    m.add_function(wrap_pyfunction!(load_precomputed_values, m)?)?;
    m.add_function(wrap_pyfunction!(main_loop, m)?)?;
    m.add_function(wrap_pyfunction!(test, m)?)?;
    Ok(())
}
