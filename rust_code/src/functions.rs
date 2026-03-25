use crate::DataHolder;
use crate::U256;

use std::cmp;
use std::env;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static EDGE_NEIGHBORHOOD_PRUNE_ENABLED: OnceLock<bool> = OnceLock::new();
static GLOBAL_COVERAGE_PRUNE_ENABLED: OnceLock<bool> = OnceLock::new();
static FACEPERM_BY_CUBE_ENABLED: OnceLock<bool> = OnceLock::new();
static PRIORITIZED_EDGE_SINGLE_PASS_ENABLED: OnceLock<bool> = OnceLock::new();
static EDGE_NEIGHBORHOOD_LOOKAHEAD_DEPTH: OnceLock<usize> = OnceLock::new();
static EDGE_NEIGHBORHOOD_LOOKAHEAD_NODE_BUDGET: OnceLock<usize> = OnceLock::new();
static RUN35_COMPAT: OnceLock<bool> = OnceLock::new();
static MAX_SQUARES: OnceLock<u32> = OnceLock::new();
static PRUNE_GLOBAL_COVERAGE: AtomicU64 = AtomicU64::new(0);
static PRUNE_EDGE_NEIGHBORHOOD: AtomicU64 = AtomicU64::new(0);
static CANON_CALLS: AtomicU64 = AtomicU64::new(0);
static CANON_ROOT_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static CANON_TAU_TRIALS: AtomicU64 = AtomicU64::new(0);

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

pub fn run35_compat_enabled() -> bool {
    *RUN35_COMPAT.get_or_init(|| env_bool("RUN35_COMPAT", false))
}

pub fn edge_neighborhood_prune_enabled() -> bool {
    if run35_compat_enabled() {
        return true;
    }
    *EDGE_NEIGHBORHOOD_PRUNE_ENABLED.get_or_init(|| {
        env_bool_negated("EDGE_NEIGHBORHOOD_PRUNE", true)
    })
}

pub fn global_coverage_prune_enabled() -> bool {
    if run35_compat_enabled() {
        return true;
    }
    *GLOBAL_COVERAGE_PRUNE_ENABLED.get_or_init(|| {
        env_bool_negated("GLOBAL_COVERAGE_PRUNE", true)
    })
}

pub fn faceperm_by_cube_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *FACEPERM_BY_CUBE_ENABLED.get_or_init(|| {
        env_bool("FACEPERM_BY_CUBE", false)
    })
}

pub fn max_squares() -> u32 {
    if run35_compat_enabled() {
        return 36;
    }
    *MAX_SQUARES.get_or_init(|| {
        match env::var("MAX_SQUARES") {
            Ok(v) => v.trim().parse::<u32>().ok().filter(|x| *x > 0).unwrap_or(36),
            Err(_) => 36,
        }
    })
}

pub fn prioritized_edge_single_pass_enabled() -> bool {
    if run35_compat_enabled() {
        return false;
    }
    *PRIORITIZED_EDGE_SINGLE_PASS_ENABLED.get_or_init(|| {
        env_bool_negated("PRIORITIZED_EDGE_SINGLE_PASS", true)
    })
}

pub fn edge_neighborhood_lookahead_depth() -> usize {
    if run35_compat_enabled() {
        return 1;
    }
    *EDGE_NEIGHBORHOOD_LOOKAHEAD_DEPTH.get_or_init(|| {
        match env::var("EDGE_NEIGHBORHOOD_LOOKAHEAD") {
            Ok(v) => v
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|d| *d > 0)
                .unwrap_or(4),
            Err(_) => 4,
        }
    })
}

pub fn edge_neighborhood_lookahead_node_budget() -> usize {
    if run35_compat_enabled() {
        return 0;
    }
    *EDGE_NEIGHBORHOOD_LOOKAHEAD_NODE_BUDGET.get_or_init(|| {
        match env::var("EDGE_NEIGHBORHOOD_LOOKAHEAD_NODE_BUDGET") {
            Ok(v) => v
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .unwrap_or(200_000),
            Err(_) => 200_000,
        }
    })
}

pub fn reset_extend_prune_counters() {
    PRUNE_GLOBAL_COVERAGE.store(0, Ordering::Relaxed);
    PRUNE_EDGE_NEIGHBORHOOD.store(0, Ordering::Relaxed);
    CANON_CALLS.store(0, Ordering::Relaxed);
    CANON_ROOT_CANDIDATES.store(0, Ordering::Relaxed);
    CANON_TAU_TRIALS.store(0, Ordering::Relaxed);
}

pub fn read_extend_prune_counters() -> (u64, u64) {
    (
        PRUNE_GLOBAL_COVERAGE.load(Ordering::Relaxed),
        PRUNE_EDGE_NEIGHBORHOOD.load(Ordering::Relaxed),
    )
}

pub fn read_canon_counters() -> (u64, u64, u64) {
    (
        CANON_CALLS.load(Ordering::Relaxed),
        CANON_ROOT_CANDIDATES.load(Ordering::Relaxed),
        CANON_TAU_TRIALS.load(Ordering::Relaxed),
    )
}

//PART I: EXTENSION

fn freeedges(ncplx: &U256, holder: &DataHolder) -> U256 {
    let mut once = U256::zero();
    let mut twice = U256::zero();
    let mut bits = *ncplx;

    while bits != U256::zero() {
        let i = bits.trailing_zeros() as usize;
        twice |= &once & &holder.boundaries[i];
        once ^= &holder.boundaries[i]; // ASSUMING THERE ARE NO EDGES INDICENT TO MORE THAN TWO SQUARES (!)
        bits ^= U256::one() << i;
    }

    once
}

fn bucket_cap(bucket: &[usize; 5], k: usize) -> usize {
    if k == 0 {
        return 0;
    }
    let mut rem = k;
    let mut cap = 0usize;
    for c in (1..=4).rev() {
        if rem == 0 {
            break;
        }
        let take = cmp::min(rem, bucket[c]);
        cap += take * c;
        rem -= take;
    }
    cap
}

fn freeedge_state_and_bucket(ncplx: &U256, holder: &DataHolder) -> (U256, U256, [usize; 5]) {
    let mut once = U256::zero();
    let mut twice = U256::zero();
    let mut bits = *ncplx;
    while bits != U256::zero() {
        let i = bits.trailing_zeros() as usize;
        let b = holder.boundaries[i];
        twice |= &once & &b;
        once ^= b;
        bits ^= U256::one() << i;
    }

    // Bucket candidate squares by how many current free edges they can cover (1..4).
    let mut bucket = [0usize; 5];
    for i in 0..*holder.n2 {
        if (ncplx & (U256::one() << i)) != U256::zero() {
            continue;
        }
        let b = holder.boundaries[i];
        // Squares hitting saturated edges are not legal future moves.
        if (b & twice) != U256::zero() {
            continue;
        }
        let c = (b & once).count_ones() as usize;
        if c > 0 {
            bucket[c] += 1;
        }
    }
    (once, twice, bucket)
}

pub fn firstone(n: U256) -> usize {
    n.trailing_zeros() as usize // ASSUMES n IS NEVER 0 (!)
}

// ANOTHER TRY:

fn map_again(input: usize) -> usize {
    let value = match input {
        // 0 => 0,   // 0  <- 00
        1 => 9,   // 17 <- 21
        2 => 8,   // 26 <- 32
        // 3 => 3,   // 27 <- 33
        4 => 7,   // 34 <- 42
        5 => 6,   // 35 <- 43
        // 6 => 6,   // 36 <- 44
        7 => 5,   // 43 <- 53
        8 => 4,   // 44 <- 54
        // 9 => 9,   // 45 <- 55
        10 => 3, // 51 <- 63
        11 => 2, // 52 <- 64
        12 => 1, // 53 <- 65
        // 13 => 13, // 54 <- 66
        _ => 0
    };
    value
}

fn prioritized_edge_mask(ncplx: &U256, t: U256, holder: &DataHolder) -> U256 {
    let dd = testvertices(ncplx, holder);
    let mask = U256::from(0b1111);
    let mut newt = U256::zero();
    if prioritized_edge_single_pass_enabled() {
        let mut vertex_deg: Vec<usize> = Vec::with_capacity(*holder.n0);
        for j in 0..*holder.n0 {
            let deg = map_again(((&dd >> (4 * j)) & &mask).to_usize().expect("Fail."));
            vertex_deg.push(deg);
        }

        let mut mindeg = 15usize;
        let mut prioritized_edges: Vec<usize> = Vec::new();
        for i in 0..*holder.n1 {
            if t & (U256::one() << i) == U256::zero() {
                continue;
            }
            let mut ideg = 15usize;
            for (j, &jdeg) in vertex_deg.iter().enumerate() {
                if holder.edgeboundaries[i] & (U256::one() << (3 * j)) != U256::zero() {
                    ideg = cmp::min(ideg, jdeg);
                }
            }
            if ideg < mindeg {
                mindeg = ideg;
                prioritized_edges.clear();
                prioritized_edges.push(i);
            } else if ideg == mindeg {
                prioritized_edges.push(i);
            }
        }
        for &i in &prioritized_edges {
            newt += U256::one() << i;
        }
    } else {
        let mut mindeg = 15usize;
        for i in 0..*holder.n1 {
            if t & (U256::one() << i) == U256::zero() {
                continue;
            }
            let mut ideg = 15usize;
            for j in 0..*holder.n0 {
                let mut jdeg = 15usize;
                if holder.edgeboundaries[i] & (U256::one() << (3 * j)) != U256::zero() {
                    jdeg = map_again(((&dd >> (4 * j)) & &mask).to_usize().expect("Fail."));
                }
                ideg = cmp::min(ideg, jdeg);
            }
            mindeg = cmp::min(mindeg, ideg);
        }
        for i in 0..*holder.n1 {
            if t & (U256::one() << i) == U256::zero() {
                continue;
            }
            let mut ideg = 15usize;
            for j in 0..*holder.n0 {
                let mut jdeg = 15usize;
                if holder.edgeboundaries[i] & (U256::one() << (3 * j)) != U256::zero() {
                    jdeg = map_again(((&dd >> (4 * j)) & &mask).to_usize().expect("Fail."));
                }
                ideg = cmp::min(ideg, jdeg);
            }
            if ideg == mindeg {
                newt += U256::one() << i;
            }
        }
    }
    newt
}

fn has_feasible_neighborhood_path(
    ncplx: &U256,
    holder: &DataHolder,
    t: U256,
    twice: U256,
    bucket: [usize; 5],
    rem: usize,
    es: U256,
    depth: usize,
) -> bool {
    if rem == 0 {
        return t == U256::zero();
    }
    let nt = t.count_ones() as usize;
    for i in 0..*holder.n2 {
        let bit = U256::one() << i;
        if (es & bit) == U256::zero() || (ncplx & bit) != U256::zero() {
            continue;
        }
        let b = holder.boundaries[i];
        if (b & twice) != U256::zero() {
            continue;
        }
        let c = (b & t).count_ones() as usize;
        if c == 0 {
            continue;
        }
        let mut bkt = bucket;
        bkt[c] -= 1;
        let rem_after = rem - 1;
        let cap_after = bucket_cap(&bkt, rem_after);
        if nt.saturating_sub(c) > cap_after {
            continue;
        }
        if depth <= 1 || rem_after == 0 {
            return true;
        }

        let ncplx1 = *ncplx | bit;
        let (t1, twice1, bucket1) = freeedge_state_and_bucket(&ncplx1, holder);
        if t1 == U256::zero() {
            return true;
        }
        let newt1 = prioritized_edge_mask(&ncplx1, t1, holder);
        if newt1 == U256::zero() {
            continue;
        }
        let es1 = holder.edgesquares[firstone(newt1)];
        if has_feasible_neighborhood_path(
            &ncplx1,
            holder,
            t1,
            twice1,
            bucket1,
            rem_after,
            es1,
            depth - 1,
        ) {
            return true;
        }
    }
    false
}

fn has_feasible_neighborhood_path_budgeted(
    ncplx: &U256,
    holder: &DataHolder,
    t: U256,
    twice: U256,
    bucket: [usize; 5],
    rem: usize,
    es: U256,
    depth: usize,
    node_budget: usize,
    nodes_visited: &mut usize,
) -> Result<bool, String> {
    *nodes_visited += 1;
    if *nodes_visited > node_budget {
        return Err(format!(
            "EDGE_NEIGHBORHOOD_LOOKAHEAD exceeded node budget: visited={} budget={} depth={} rem={} ns={}",
            *nodes_visited,
            node_budget,
            depth,
            rem,
            ncplx.count_ones()
        ));
    }

    if rem == 0 {
        return Ok(t == U256::zero());
    }
    let nt = t.count_ones() as usize;
    for i in 0..*holder.n2 {
        let bit = U256::one() << i;
        if (es & bit) == U256::zero() || (ncplx & bit) != U256::zero() {
            continue;
        }
        let b = holder.boundaries[i];
        if (b & twice) != U256::zero() {
            continue;
        }
        let c = (b & t).count_ones() as usize;
        if c == 0 {
            continue;
        }
        let mut bkt = bucket;
        bkt[c] -= 1;
        let rem_after = rem - 1;
        let cap_after = bucket_cap(&bkt, rem_after);
        if nt.saturating_sub(c) > cap_after {
            continue;
        }
        if depth <= 1 || rem_after == 0 {
            return Ok(true);
        }

        let ncplx1 = *ncplx | bit;
        let (t1, twice1, bucket1) = freeedge_state_and_bucket(&ncplx1, holder);
        if t1 == U256::zero() {
            return Ok(true);
        }
        let newt1 = prioritized_edge_mask(&ncplx1, t1, holder);
        if newt1 == U256::zero() {
            continue;
        }
        let es1 = holder.edgesquares[firstone(newt1)];
        if has_feasible_neighborhood_path_budgeted(
            &ncplx1,
            holder,
            t1,
            twice1,
            bucket1,
            rem_after,
            es1,
            depth - 1,
            node_budget,
            nodes_visited,
        )? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn anotherextendonce(ncplx: &U256, holder: &DataHolder) -> Vec<U256> {
    let mut ncplxsnew = vec![];
    let (t, twice, bucket) = freeedge_state_and_bucket(ncplx, holder);

    let ns = ncplx.count_ones();
    let max_sq = max_squares();
    if ns >= max_sq {
        return ncplxsnew;
    }
    let nt = t.count_ones();
    let rem_u32 = max_sq - ns;
    let rem = rem_u32 as usize;
    if nt <= 4 * rem_u32 {
        // Stronger necessary condition: even optimally chosen future squares must be
        // able to touch all currently free edges within the remaining move budget.
        if global_coverage_prune_enabled() {
            if (nt as usize) > bucket_cap(&bucket, rem) {
                PRUNE_GLOBAL_COVERAGE.fetch_add(1, Ordering::Relaxed);
                return ncplxsnew;
            }
        }
        let newt = prioritized_edge_mask(ncplx, t, holder);
        if newt == U256::zero() {
            return ncplxsnew;
        }
        let es = holder.edgesquares[firstone(newt)];

        // Safe strengthening on the prioritized neighborhood:
        // at least one admissible first move from `es` must leave enough
        // theoretical coverage capacity for the remaining `rem - 1` moves.
        if rem > 0 && edge_neighborhood_prune_enabled() {
            let lookahead_depth = edge_neighborhood_lookahead_depth();
            let any_feasible = if lookahead_depth <= 3 {
                has_feasible_neighborhood_path(
                    ncplx,
                    holder,
                    t,
                    twice,
                    bucket,
                    rem,
                    es,
                    lookahead_depth,
                )
            } else {
                let node_budget = edge_neighborhood_lookahead_node_budget();
                let mut nodes_visited = 0usize;
                match has_feasible_neighborhood_path_budgeted(
                    ncplx,
                    holder,
                    t,
                    twice,
                    bucket,
                    rem,
                    es,
                    lookahead_depth,
                    node_budget,
                    &mut nodes_visited,
                ) {
                    Ok(v) => v,
                    Err(msg) => panic!(
                        "{} (lookahead={} node_budget={} ns={} nt={} rem={})",
                        msg,
                        lookahead_depth,
                        node_budget,
                        ns,
                        nt,
                        rem
                    ),
                }
            };
            if !any_feasible {
                PRUNE_EDGE_NEIGHBORHOOD.fetch_add(1, Ordering::Relaxed);
                return ncplxsnew;
            }
        }

        for i in 0..*holder.n2 {
            if es & (U256::one() << i) != U256::zero()
                && ncplx & (U256::one() << i) == U256::zero()
                && (holder.boundaries[i] & twice) == U256::zero()
            {
                ncplxsnew.push(ncplx|(U256::one()<<i))
            }
        }
    }

    ncplxsnew
}

pub fn disconnected_withbdry_extendonce(ncplx: &U256, holder: &DataHolder) -> Vec<U256> {
    let mut ncplxsnew = vec![];

    for i in 0..*holder.n2 {
        if ncplx & (U256::one() << i) == U256::zero() {
            ncplxsnew.push(ncplx|(U256::one()<<i))
        }
    }

    ncplxsnew
}

pub fn withbdry_extendonce(ncplx: &U256, holder: &DataHolder) -> Vec<U256> {
    let mut ncplxsnew = vec![];
    let mut es = U256::zero();
    let t = freeedges(ncplx,&holder);

    for i in 0..*holder.n1 {
        if (&t >> i) & U256::one() != U256::zero() {
            es |= &holder.edgesquares[i]
        }
    }

    for i in 0..*holder.n2 {
        if &es & (U256::one() << i) != U256::zero() && ncplx & (U256::one() << i) == U256::zero() {
            ncplxsnew.push(ncplx|(U256::one()<<i))
        }
    }

    ncplxsnew
}

pub fn disconnected_extendonce(ncplx: &U256, holder: &DataHolder) -> Vec<U256> {
    let mut ncplxsnew = vec![];
    let t = freeedges(ncplx,&holder);

    if t == U256::zero() {
        for i in 0..*holder.n2 {
            if ncplx & (U256::one() << i) == U256::zero() {
                ncplxsnew.push(ncplx|(U256::one()<<i))
            }
        }

        return ncplxsnew;
    }

    let es = &holder.edgesquares[firstone(t)];

    for i in 0..*holder.n2 {
        if es & (U256::one() << i) != U256::zero() && ncplx & (U256::one() << i) == U256::zero() {
            ncplxsnew.push(ncplx|(U256::one()<<i))
        }
    }

    ncplxsnew
}

// PART II: CANONICAL LABELING

fn vec_to_octal(vect: &[usize]) -> usize {
    let mut result = 0;
    let mut pow = 0;

    for &digit in vect.iter().rev() {
        result |= digit << pow;
        pow += 3;
    }

    result
}

fn neighborhood_pattern(dd: &[Vec<usize>], v: usize, holder: &DataHolder) -> usize {
    let values: Vec<&Vec<usize>> = holder.nnbhd[v].iter().map(|u| &dd[*u]).collect();
    let mut unique_sorted = values.clone();
    unique_sorted.sort_unstable();
    unique_sorted.dedup();

    let mut encoded = Vec::with_capacity(values.len());
    for value in values {
        let rank = unique_sorted.binary_search(&value).expect("Fail.");
        encoded.push(rank);
    }
    vec_to_octal(&encoded)
}

fn alledges(ncplx: &U256, holder: &DataHolder) -> U256 {
    let mut edges = U256::zero();
    let mut bits = *ncplx;

    while bits != U256::zero() {
        let i = bits.trailing_zeros() as usize;
        edges |= &holder.boundaries[i];
        bits ^= U256::one() << i;
    }

    edges
}

fn map_to_range(input: usize) -> U256 {
    let value = match input {
        0 => 0,
        17 => 1,
        26 => 2,
        27 => 3,
        34 => 4,
        35 => 5,
        36 => 6,
        43 => 7,
        44 => 8,
        45 => 9,
        51 => 10,
        52 => 11,
        53 => 12,
        54 => 13,
        _ => 14 // USED TO BE 0
    };
    U256::from(value)
}

fn u256intertwine(mut x: U256, mut y: U256, holder: &DataHolder) -> U256 {
    let mut result = U256::zero();
    let mut pow = 0;

    for _ in 0..*holder.n0 {
        let element = ((x & U256::from(0b111)) << 3) | (y & U256::from(0b111));
        result |= map_to_range(element.to_usize().expect("Fail."))<<pow;
        pow += 4;
        x >>= 3;
        y >>= 3;
    }

    result
}

fn testvertices(ncplx: &U256, holder: &DataHolder) -> U256 {
    let edges = alledges(ncplx,&holder);

    let mut vine = U256::zero();
    let mut edge_bits = edges;
    while edge_bits != U256::zero() {
        let i = edge_bits.trailing_zeros() as usize;
        vine += holder.edgeboundaries[i];
        edge_bits ^= U256::one() << i;
    }

    let mut vins = U256::zero();
    let mut sq_bits = *ncplx;
    while sq_bits != U256::zero() {
        let i = sq_bits.trailing_zeros() as usize;
        vins += holder.bboundaries[i];
        sq_bits ^= U256::one() << i;
    }

    u256intertwine(vine, vins, &holder)
}

pub fn vertex_codes(ncplx: &U256, holder: &DataHolder) -> Vec<u8> {
    let mut dd = testvertices(ncplx, holder);
    let mask = U256::from(0b1111);
    let mut codes = Vec::with_capacity(*holder.n0);
    for _ in 0..*holder.n0 {
        codes.push((&dd & &mask).to_usize().expect("Fail.") as u8);
        dd >>= 4;
    }
    codes
}

fn newbigdegs_from_codes(dd: &[u8], holder: &DataHolder) -> Vec<usize> {
    let mut lst = Vec::with_capacity(*holder.n0);

    for v in 0..*holder.n0 {
        let mut elt = dd[v] as usize;
        let mut t_values: Vec<u8> = Vec::with_capacity(holder.nnbhd[v].len());

        for u in &holder.nnbhd[v] {
            t_values.push(dd[*u]);
        }
        t_values.sort_unstable();

        for t in t_values {
            elt = (elt << 4) | (t as usize);
        }

        lst.push(elt);
    }

    lst
}

fn bigbigdegs_from_codes(dd: &[u8], holder: &DataHolder) -> Vec<Vec<usize>> {
    let dd = newbigdegs_from_codes(dd, holder);
    let mut result = Vec::with_capacity(*holder.n0);

    for v in 0..*holder.n0 {
        let mut u_values = Vec::with_capacity(holder.nnbhd[v].len());

        for u in &holder.nnbhd[v] {
            u_values.push(dd[*u]);
        }
        u_values.sort();

        let mut component = Vec::with_capacity(u_values.len() + 1);
        component.push(dd[v]);
        component.extend(u_values);

        result.push(component);
    }

    result
}

fn bigbigdegs(ncplx: &U256, holder: &DataHolder) -> Vec<Vec<usize>> {
    let dd = vertex_codes(ncplx, holder);
    bigbigdegs_from_codes(&dd, holder)
}

fn apply(perm: &[usize], ncplx: &U256) -> U256 {
    let mut permcplx = U256::zero();
    let mut bits = *ncplx;

    while bits != U256::zero() {
        let i = bits.trailing_zeros() as usize;
        permcplx |= U256::one() << perm[i];
        bits ^= U256::one() << i;
    }

    permcplx
}

pub fn cubicalcanlabel(ncplx: &U256, holder: &DataHolder) -> U256 {
    let dd = bigbigdegs(ncplx, holder);
    let mindeg = dd.iter().min().unwrap();
    let mut best: Option<U256> = None;
    let use_by_cube = faceperm_by_cube_enabled();
    let mut root_candidates = 0u64;
    let mut tau_trials = 0u64;
    CANON_CALLS.fetch_add(1, Ordering::Relaxed);

    for v in 0..*holder.n0 {
        if dd[v] != *mindeg {
            continue;
        }
        root_candidates += 1;
        let nbdegs = neighborhood_pattern(&dd, v, holder);
        let cube_v = holder.cubes[v].clone();
        for tau in &holder.permlist[nbdegs] {
            tau_trials += 1;
            let faceperm = if use_by_cube {
                holder.get_faceperm_for(v, tau)
            } else {
                let key = (cube_v.clone(), tau.clone());
                holder.get_faceperm(&key)
            };
            let cand = apply(faceperm, ncplx);
            if best.as_ref().map_or(true, |b| cand < *b) {
                best = Some(cand);
            }
        }
    }

    CANON_ROOT_CANDIDATES.fetch_add(root_candidates, Ordering::Relaxed);
    CANON_TAU_TRIALS.fetch_add(tau_trials, Ordering::Relaxed);

    best.unwrap_or_else(U256::zero)
}

pub fn cubicalcanlabel_from_codes(ncplx: &U256, dd_codes: &[u8], holder: &DataHolder) -> U256 {
    let dd = bigbigdegs_from_codes(dd_codes,holder);
    let mindeg = dd.iter().min().unwrap();
    let mut best: Option<U256> = None;
    let use_by_cube = faceperm_by_cube_enabled();
    let mut root_candidates = 0u64;
    let mut tau_trials = 0u64;
    CANON_CALLS.fetch_add(1, Ordering::Relaxed);

    for v in 0..*holder.n0 {
        if dd[v] != *mindeg {
            continue;
        }
        root_candidates += 1;
        let nbdegs = neighborhood_pattern(&dd, v, holder);
        let cube_v = holder.cubes[v].clone();
        for tau in &holder.permlist[nbdegs] {
            tau_trials += 1;
            let faceperm = if use_by_cube {
                holder.get_faceperm_for(v, tau)
            } else {
                let key = (cube_v.clone(), tau.clone());
                holder.get_faceperm(&key)
            };
            let cand = apply(faceperm, ncplx);
            if best.as_ref().map_or(true, |b| cand < *b) {
                best = Some(cand);
            }
        }
    }

    CANON_ROOT_CANDIDATES.fetch_add(root_candidates, Ordering::Relaxed);
    CANON_TAU_TRIALS.fetch_add(tau_trials, Ordering::Relaxed);

    best.unwrap_or_else(U256::zero)
}

pub fn nolabel(ncplx: &U256, _holder: &DataHolder) -> U256 {
    ncplx.clone()
}

// PART III: TESTING

pub fn testedges(ncplx: &U256, holder: &DataHolder) -> (bool, bool) {
    let mut once = U256::zero();
    let mut twice = U256::zero();
    let mut bits = *ncplx;

    while bits != U256::zero() {
        let i = bits.trailing_zeros() as usize;
        twice |= &once & &holder.boundaries[i];
        once ^= &holder.boundaries[i];
        if (&once & &twice) != U256::zero() { //PREVERI!!
            return (false, false);
        }
        bits ^= U256::one() << i;
    }

    (true, once == U256::zero())
}

// PART IV: LINK CONDITION

pub fn badverticesq(ncplx: &U256, holder: &DataHolder) -> bool {
    bad_and_reallybad(ncplx, holder).0
}

pub fn bad_and_reallybad_from_codes(ncplx: &U256, dd: &[u8], holder: &DataHolder) -> (bool, bool) {
    let mut bad = false;
    let mut reallybad = false;

    for v in 0..*holder.n0 {
        let ddv = dd[v] as usize;
        if ddv == 4 || ddv == 7 || ddv == 10 || ddv == 11 {
            bad = true;
        }
        if ddv == 14 {
            reallybad = true; // SHOULDN'T HAPPEN, BUT ADDED ANYWAY.
            break;
        }
        if ddv == 8 || ddv == 12 || ddv == 13 {
            for cyc in &holder.cycles3[v] {
                if cyc & ncplx == *cyc {
                    reallybad = true;
                    break;
                }
            }
        }
        if !reallybad && ddv == 12 {
            for cyc in &holder.cycles4[v] {
                if cyc & ncplx == *cyc {
                    reallybad = true;
                    break;
                }
            }
        }
        if bad && reallybad {
            break;
        }
    }
    (bad, reallybad)
}

pub fn bad_and_reallybad(ncplx: &U256, holder: &DataHolder) -> (bool, bool) {
    let dd = vertex_codes(ncplx, holder);
    bad_and_reallybad_from_codes(ncplx, &dd, holder)
}

pub fn reallybadverticesq(ncplx: &U256, holder: &DataHolder) -> bool {
    bad_and_reallybad(ncplx, holder).1
}
