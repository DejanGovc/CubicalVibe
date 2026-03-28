use crate::DataHolder;
use crate::u256::BitBackend;

use std::cmp;
use std::env;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static EDGE_NEIGHBORHOOD_PRUNE_ENABLED: OnceLock<bool> = OnceLock::new();
static GLOBAL_COVERAGE_PRUNE_ENABLED: OnceLock<bool> = OnceLock::new();
static PRIORITIZED_EDGE_SINGLE_PASS_ENABLED: OnceLock<bool> = OnceLock::new();
static EDGE_NEIGHBORHOOD_LOOKAHEAD_DEPTH: OnceLock<usize> = OnceLock::new();
static EDGE_NEIGHBORHOOD_LOOKAHEAD_NODE_BUDGET: OnceLock<usize> = OnceLock::new();
static RUN35_COMPAT: OnceLock<bool> = OnceLock::new();
static MAX_SQUARES: OnceLock<Option<u32>> = OnceLock::new();
static CANON_THREE_HOP_REFINEMENT: OnceLock<bool> = OnceLock::new();
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

pub fn max_squares() -> Option<u32> {
    if run35_compat_enabled() {
        return Some(36);
    }
    *MAX_SQUARES.get_or_init(|| {
        match env::var("MAX_SQUARES") {
            Ok(v) => v.trim().parse::<u32>().ok().filter(|x| *x > 0),
            Err(_) => None,
        }
    })
}

pub fn canon_three_hop_refinement_enabled() -> bool {
    *CANON_THREE_HOP_REFINEMENT.get_or_init(|| {
        env_bool("CANON_THREE_HOP_REFINEMENT", false)
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

fn freeedges<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> B {
    let mut once = B::zero();
    let mut twice = B::zero();
    let mut bits = *ncplx;

    while bits != B::zero() {
        let i = bits.trailing_zeros() as usize;
        twice |= once & holder.boundaries[i];
        once ^= holder.boundaries[i]; // ASSUMING THERE ARE NO EDGES INDICENT TO MORE THAN TWO SQUARES (!)
        bits ^= B::one() << i;
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

fn freeedge_state_and_bucket<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> (B, B, [usize; 5]) {
    let mut once = B::zero();
    let mut twice = B::zero();
    let mut bits = *ncplx;
    while bits != B::zero() {
        let i = bits.trailing_zeros() as usize;
        let b = holder.boundaries[i];
        twice |= once & b;
        once ^= b;
        bits ^= B::one() << i;
    }

    // Bucket candidate squares by how many current free edges they can cover (1..4).
    let mut bucket = [0usize; 5];
    for i in 0..*holder.n2 {
        if (*ncplx & (B::one() << i)) != B::zero() {
            continue;
        }
        let b = holder.boundaries[i];
        // Squares hitting saturated edges are not legal future moves.
        if (b & twice) != B::zero() {
            continue;
        }
        let c = (b & once).count_ones() as usize;
        if c > 0 {
            bucket[c] += 1;
        }
    }
    (once, twice, bucket)
}

pub fn firstone<B: BitBackend>(n: B) -> usize {
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

fn prioritized_edge_mask<B: BitBackend>(ncplx: &B, t: B, holder: &DataHolder<B>) -> B {
    let dd = testvertices(ncplx, holder);
    let mask = B::from(0b1111u8);
    let mut newt = B::zero();
    if prioritized_edge_single_pass_enabled() {
        let mut vertex_deg: Vec<usize> = Vec::with_capacity(*holder.n0);
        for j in 0..*holder.n0 {
            let deg = map_again(((dd >> (4 * j)) & mask).to_usize().expect("Fail."));
            vertex_deg.push(deg);
        }

        let mut mindeg = 15usize;
        let mut prioritized_edges: Vec<usize> = Vec::new();
        for i in 0..*holder.n1 {
            if t & (B::one() << i) == B::zero() {
                continue;
            }
            let mut ideg = 15usize;
            for (j, &jdeg) in vertex_deg.iter().enumerate() {
                if holder.edgeboundaries[i] & (B::one() << (3 * j)) != B::zero() {
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
            newt += B::one() << i;
        }
    } else {
        let mut mindeg = 15usize;
        for i in 0..*holder.n1 {
            if t & (B::one() << i) == B::zero() {
                continue;
            }
            let mut ideg = 15usize;
            for j in 0..*holder.n0 {
                let mut jdeg = 15usize;
                if holder.edgeboundaries[i] & (B::one() << (3 * j)) != B::zero() {
                    jdeg = map_again(((dd >> (4 * j)) & mask).to_usize().expect("Fail."));
                }
                ideg = cmp::min(ideg, jdeg);
            }
            mindeg = cmp::min(mindeg, ideg);
        }
        for i in 0..*holder.n1 {
            if t & (B::one() << i) == B::zero() {
                continue;
            }
            let mut ideg = 15usize;
            for j in 0..*holder.n0 {
                let mut jdeg = 15usize;
                if holder.edgeboundaries[i] & (B::one() << (3 * j)) != B::zero() {
                    jdeg = map_again(((dd >> (4 * j)) & mask).to_usize().expect("Fail."));
                }
                ideg = cmp::min(ideg, jdeg);
            }
            if ideg == mindeg {
                newt += B::one() << i;
            }
        }
    }
    newt
}

fn has_feasible_neighborhood_path<B: BitBackend>(
    ncplx: &B,
    holder: &DataHolder<B>,
    t: B,
    twice: B,
    bucket: [usize; 5],
    rem: usize,
    es: B,
    depth: usize,
) -> bool {
    if rem == 0 {
        return t == B::zero();
    }
    let nt = t.count_ones() as usize;
    for i in 0..*holder.n2 {
        let bit = B::one() << i;
        if (es & bit) == B::zero() || (*ncplx & bit) != B::zero() {
            continue;
        }
        let b = holder.boundaries[i];
        if (b & twice) != B::zero() {
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
        if t1 == B::zero() {
            return true;
        }
        let newt1 = prioritized_edge_mask(&ncplx1, t1, holder);
        if newt1 == B::zero() {
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

fn has_feasible_neighborhood_path_budgeted<B: BitBackend>(
    ncplx: &B,
    holder: &DataHolder<B>,
    t: B,
    twice: B,
    bucket: [usize; 5],
    rem: usize,
    es: B,
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
        return Ok(t == B::zero());
    }
    let nt = t.count_ones() as usize;
    for i in 0..*holder.n2 {
        let bit = B::one() << i;
        if (es & bit) == B::zero() || (*ncplx & bit) != B::zero() {
            continue;
        }
        let b = holder.boundaries[i];
        if (b & twice) != B::zero() {
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
        if t1 == B::zero() {
            return Ok(true);
        }
        let newt1 = prioritized_edge_mask(&ncplx1, t1, holder);
        if newt1 == B::zero() {
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

pub fn anotherextendonce<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<B> {
    let mut ncplxsnew = vec![];
    let (t, twice, bucket) = freeedge_state_and_bucket(ncplx, holder);

    let ns = ncplx.count_ones();
    let max_sq = max_squares().unwrap_or(*holder.n2 as u32);
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
        if newt == B::zero() {
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
            if es & (B::one() << i) != B::zero()
                && *ncplx & (B::one() << i) == B::zero()
                && (holder.boundaries[i] & twice) == B::zero()
            {
                ncplxsnew.push(*ncplx | (B::one() << i))
            }
        }
    }

    ncplxsnew
}

pub fn disconnected_withbdry_extendonce<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<B> {
    let mut ncplxsnew = vec![];

    for i in 0..*holder.n2 {
        if *ncplx & (B::one() << i) == B::zero() {
            ncplxsnew.push(*ncplx | (B::one() << i))
        }
    }

    ncplxsnew
}

pub fn withbdry_extendonce<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<B> {
    let mut ncplxsnew = vec![];
    let mut es = B::zero();
    let t = freeedges(ncplx,&holder);

    for i in 0..*holder.n1 {
        if (t >> i) & B::one() != B::zero() {
            es |= holder.edgesquares[i]
        }
    }

    for i in 0..*holder.n2 {
        if es & (B::one() << i) != B::zero() && *ncplx & (B::one() << i) == B::zero() {
            ncplxsnew.push(*ncplx | (B::one() << i))
        }
    }

    ncplxsnew
}

pub fn disconnected_extendonce<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<B> {
    let mut ncplxsnew = vec![];
    let t = freeedges(ncplx,&holder);

    if t == B::zero() {
        for i in 0..*holder.n2 {
            if *ncplx & (B::one() << i) == B::zero() {
                ncplxsnew.push(*ncplx | (B::one() << i))
            }
        }

        return ncplxsnew;
    }

    let es = holder.edgesquares[firstone(t)];

    for i in 0..*holder.n2 {
        if es & (B::one() << i) != B::zero() && *ncplx & (B::one() << i) == B::zero() {
            ncplxsnew.push(*ncplx | (B::one() << i))
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

fn neighborhood_pattern<T: Ord, B: BitBackend>(dd: &[T], v: usize, holder: &DataHolder<B>) -> usize {
    let values: Vec<&T> = holder.nnbhd[v].iter().map(|u| &dd[*u]).collect();
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

fn alledges<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> B {
    let mut edges = B::zero();
    let mut bits = *ncplx;

    while bits != B::zero() {
        let i = bits.trailing_zeros() as usize;
        edges |= holder.boundaries[i];
        bits ^= B::one() << i;
    }

    edges
}

fn vertex_code_bits<B: BitBackend>(holder: &DataHolder<B>) -> usize {
    if *holder.n_dim >= 7 { 5 } else { 4 }
}

fn map_to_range<B: BitBackend>(input: usize, holder: &DataHolder<B>) -> B {
    let value = if *holder.n_dim >= 7 {
        match input {
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
            60 => 14, // (7,4)
            61 => 15, // (7,5)
            62 => 16, // (7,6)
            63 => 17, // (7,7)
            _ => 18,
        }
    } else {
        match input {
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
            _ => 14,
        }
    };
    B::from(value as u128)
}

fn intertwine_vertex_codes<B: BitBackend>(mut x: B, mut y: B, holder: &DataHolder<B>) -> B {
    let mut result = B::zero();
    let mut pow = 0;
    let bits = vertex_code_bits(holder);

    for _ in 0..*holder.n0 {
        let element = ((x & B::from(0b111u8)) << 3) | (y & B::from(0b111u8));
        result |= map_to_range::<B>(element.to_usize().expect("Fail."), holder) << pow;
        pow += bits;
        x >>= 3;
        y >>= 3;
    }

    result
}

fn testvertices<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> B {
    let edges = alledges(ncplx,&holder);

    let mut vine = B::zero();
    let mut edge_bits = edges;
    while edge_bits != B::zero() {
        let i = edge_bits.trailing_zeros() as usize;
        vine += holder.edgeboundaries[i];
        edge_bits ^= B::one() << i;
    }

    let mut vins = B::zero();
    let mut sq_bits = *ncplx;
    while sq_bits != B::zero() {
        let i = sq_bits.trailing_zeros() as usize;
        vins += holder.bboundaries[i];
        sq_bits ^= B::one() << i;
    }

    intertwine_vertex_codes(vine, vins, &holder)
}

pub fn vertex_codes<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<u8> {
    let mut dd = testvertices(ncplx, holder);
    let bits = vertex_code_bits(holder);
    let mask = B::from(((1u8 << bits) - 1) as u128);
    let mut codes = Vec::with_capacity(*holder.n0);
    for _ in 0..*holder.n0 {
        codes.push((dd & mask).to_usize().expect("Fail.") as u8);
        dd >>= bits;
    }
    codes
}

fn newbigdegs_from_codes<B: BitBackend>(dd: &[u8], holder: &DataHolder<B>) -> Vec<usize> {
    let mut lst = Vec::with_capacity(*holder.n0);
    let bits = vertex_code_bits(holder);

    for v in 0..*holder.n0 {
        let mut elt = dd[v] as usize;
        let mut t_values: Vec<u8> = Vec::with_capacity(holder.nnbhd[v].len());

        for u in &holder.nnbhd[v] {
            t_values.push(dd[*u]);
        }
        t_values.sort_unstable();

        for t in t_values {
            elt = (elt << bits) | (t as usize);
        }

        lst.push(elt);
    }

    lst
}

fn bigbigdegs_from_codes<B: BitBackend>(dd: &[u8], holder: &DataHolder<B>) -> Vec<Vec<usize>> {
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

fn bigbigdegs<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<Vec<usize>> {
    let dd = vertex_codes(ncplx, holder);
    bigbigdegs_from_codes(&dd, holder)
}

fn refine_rank_labels<B: BitBackend>(labels: &[u8], holder: &DataHolder<B>) -> Vec<u8> {
    let mut signatures: Vec<[u8; 8]> = Vec::with_capacity(*holder.n0);

    for v in 0..*holder.n0 {
        let mut signature = [0u8; 8];
        signature[0] = labels[v];
        let mut degree = 0usize;
        for u in &holder.nnbhd[v] {
            signature[degree + 1] = labels[*u];
            degree += 1;
        }
        signature[1..=degree].sort_unstable();
        signatures.push(signature);
    }

    let mut unique_sorted = signatures.clone();
    unique_sorted.sort_unstable();
    unique_sorted.dedup();

    signatures
        .into_iter()
        .map(|signature| {
            unique_sorted
                .binary_search(&signature)
                .expect("canonical refinement signature missing") as u8
        })
        .collect()
}

fn bigbigbigdegs_from_codes<B: BitBackend>(dd: &[u8], holder: &DataHolder<B>) -> Vec<u8> {
    let bigdegs = refine_rank_labels(dd, holder);
    let bigbigdegs = refine_rank_labels(&bigdegs, holder);
    refine_rank_labels(&bigbigdegs, holder)
}

fn bigbigbigdegs<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> Vec<u8> {
    let dd = vertex_codes(ncplx, holder);
    bigbigbigdegs_from_codes(&dd, holder)
}

fn apply<B: BitBackend>(perm: &[usize], ncplx: &B) -> B {
    let mut permcplx = B::zero();
    let mut bits = *ncplx;

    while bits != B::zero() {
        let i = bits.trailing_zeros() as usize;
        permcplx |= B::one() << perm[i];
        bits ^= B::one() << i;
    }

    permcplx
}

pub fn cubicalcanlabel<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> B {
    if *holder.n_dim >= 7 && canon_three_hop_refinement_enabled() {
        let dd = bigbigbigdegs(ncplx, holder);
        let mindeg = dd.iter().min().unwrap();
        let mut best: Option<B> = None;
        let mut root_candidates = 0u64;
        let mut tau_trials = 0u64;
        CANON_CALLS.fetch_add(1, Ordering::Relaxed);

        for v in 0..*holder.n0 {
            if dd[v] != *mindeg {
                continue;
            }
            root_candidates += 1;
            let nbdegs = neighborhood_pattern(&dd, v, holder);
            let tau_list = holder.get_permlist(nbdegs, holder.nnbhd[v].len());
            for tau in tau_list.iter() {
                tau_trials += 1;
                let faceperm = holder.get_faceperm_for(v, tau);
                let cand = apply(faceperm.as_ref(), ncplx);
                if best.as_ref().map_or(true, |b| cand < *b) {
                    best = Some(cand);
                }
            }
        }

        CANON_ROOT_CANDIDATES.fetch_add(root_candidates, Ordering::Relaxed);
        CANON_TAU_TRIALS.fetch_add(tau_trials, Ordering::Relaxed);

        best.unwrap_or_else(B::zero)
    } else {
        let dd = bigbigdegs(ncplx, holder);
        let mindeg = dd.iter().min().unwrap();
        let mut best: Option<B> = None;
        let mut root_candidates = 0u64;
        let mut tau_trials = 0u64;
        CANON_CALLS.fetch_add(1, Ordering::Relaxed);

        for v in 0..*holder.n0 {
            if dd[v] != *mindeg {
                continue;
            }
            root_candidates += 1;
            let nbdegs = neighborhood_pattern(&dd, v, holder);
            let tau_list = holder.get_permlist(nbdegs, holder.nnbhd[v].len());
            for tau in tau_list.iter() {
                tau_trials += 1;
                let faceperm = holder.get_faceperm_for(v, tau);
                let cand = apply(faceperm.as_ref(), ncplx);
                if best.as_ref().map_or(true, |b| cand < *b) {
                    best = Some(cand);
                }
            }
        }

        CANON_ROOT_CANDIDATES.fetch_add(root_candidates, Ordering::Relaxed);
        CANON_TAU_TRIALS.fetch_add(tau_trials, Ordering::Relaxed);

        best.unwrap_or_else(B::zero)
    }
}

pub fn cubicalcanlabel_from_codes<B: BitBackend>(ncplx: &B, dd_codes: &[u8], holder: &DataHolder<B>) -> B {
    if *holder.n_dim >= 7 && canon_three_hop_refinement_enabled() {
        let dd = bigbigbigdegs_from_codes(dd_codes, holder);
        let mindeg = dd.iter().min().unwrap();
        let mut best: Option<B> = None;
        let mut root_candidates = 0u64;
        let mut tau_trials = 0u64;
        CANON_CALLS.fetch_add(1, Ordering::Relaxed);

        for v in 0..*holder.n0 {
            if dd[v] != *mindeg {
                continue;
            }
            root_candidates += 1;
            let nbdegs = neighborhood_pattern(&dd, v, holder);
            let tau_list = holder.get_permlist(nbdegs, holder.nnbhd[v].len());
            for tau in tau_list.iter() {
                tau_trials += 1;
                let faceperm = holder.get_faceperm_for(v, tau);
                let cand = apply(faceperm.as_ref(), ncplx);
                if best.as_ref().map_or(true, |b| cand < *b) {
                    best = Some(cand);
                }
            }
        }

        CANON_ROOT_CANDIDATES.fetch_add(root_candidates, Ordering::Relaxed);
        CANON_TAU_TRIALS.fetch_add(tau_trials, Ordering::Relaxed);

        best.unwrap_or_else(B::zero)
    } else {
        let dd = bigbigdegs_from_codes(dd_codes,holder);
        let mindeg = dd.iter().min().unwrap();
        let mut best: Option<B> = None;
        let mut root_candidates = 0u64;
        let mut tau_trials = 0u64;
        CANON_CALLS.fetch_add(1, Ordering::Relaxed);

        for v in 0..*holder.n0 {
            if dd[v] != *mindeg {
                continue;
            }
            root_candidates += 1;
            let nbdegs = neighborhood_pattern(&dd, v, holder);
            let tau_list = holder.get_permlist(nbdegs, holder.nnbhd[v].len());
            for tau in tau_list.iter() {
                tau_trials += 1;
                let faceperm = holder.get_faceperm_for(v, tau);
                let cand = apply(faceperm.as_ref(), ncplx);
                if best.as_ref().map_or(true, |b| cand < *b) {
                    best = Some(cand);
                }
            }
        }

        CANON_ROOT_CANDIDATES.fetch_add(root_candidates, Ordering::Relaxed);
        CANON_TAU_TRIALS.fetch_add(tau_trials, Ordering::Relaxed);

        best.unwrap_or_else(B::zero)
    }
}

pub fn nolabel<B: BitBackend>(ncplx: &B, _holder: &DataHolder<B>) -> B {
    ncplx.clone()
}

// PART III: TESTING

pub fn testedges<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> (bool, bool) {
    let mut once = B::zero();
    let mut twice = B::zero();
    let mut bits = *ncplx;

    while bits != B::zero() {
        let i = bits.trailing_zeros() as usize;
        twice |= once & holder.boundaries[i];
        once ^= holder.boundaries[i];
        if (once & twice) != B::zero() { //PREVERI!!
            return (false, false);
        }
        bits ^= B::one() << i;
    }

    (true, once == B::zero())
}

// PART IV: LINK CONDITION

pub fn badverticesq<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> bool {
    bad_and_reallybad(ncplx, holder).0
}

pub fn bad_and_reallybad_from_codes<B: BitBackend>(ncplx: &B, dd: &[u8], holder: &DataHolder<B>) -> (bool, bool) {
    let mut bad = false;
    let mut reallybad = false;
    let n7 = *holder.n_dim >= 7;
    let catchall_code = if n7 { 18 } else { 14 };

    for v in 0..*holder.n0 {
        let ddv = dd[v] as usize;
        if ddv == 4 || ddv == 7 || ddv == 10 || ddv == 11 || (n7 && (ddv == 14 || ddv == 15)) {
            bad = true;
        }
        if ddv == catchall_code {
            reallybad = true; // SHOULDN'T HAPPEN, BUT ADDED ANYWAY.
            break;
        }
        if ddv == 8 || ddv == 12 || ddv == 13 || (n7 && (ddv == 16 || ddv == 17)) {
            for cyc in &holder.cycles3[v] {
                if *cyc & *ncplx == *cyc {
                    reallybad = true;
                    break;
                }
            }
        }
        if !reallybad && (ddv == 12 || (n7 && (ddv == 16 || ddv == 17))) {
            for cyc in &holder.cycles4[v] {
                if *cyc & *ncplx == *cyc {
                    reallybad = true;
                    break;
                }
            }
        }
        if !reallybad && n7 && ddv == 16 {
            for cyc in &holder.cycles5[v] {
                if *cyc & *ncplx == *cyc {
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

pub fn bad_and_reallybad<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> (bool, bool) {
    let dd = vertex_codes(ncplx, holder);
    bad_and_reallybad_from_codes(ncplx, &dd, holder)
}

pub fn reallybadverticesq<B: BitBackend>(ncplx: &B, holder: &DataHolder<B>) -> bool {
    bad_and_reallybad(ncplx, holder).1
}
