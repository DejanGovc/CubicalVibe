from functools import lru_cache
from itertools import product
from itertools import groupby
from itertools import permutations
from itertools import combinations
import os
import pickle
import time

import rust_code

### SETTING GLOBAL VARIABLES

n = None
n0 = None
n1 = None
n2 = None
chunksize = None
num_bytes = None
PRECOMPUTE_CACHE_VERSION = 3

def initialize_globals(N,CHUNKSIZE):
    global n, n2, n1, n0, chunksize, num_bytes
    n = N
    chunksize=CHUNKSIZE
    n2 = n*(n-1)*2**(n-3)
    n1 = n*2**(n-1)
    n0 = 2**n
    num_bytes = n2//8 #8 bits per byte
    if N == 3:
        num_bytes = 1

### FORMATS FOR SQLITE3:

def int_to_blob(val):
    return val.to_bytes(num_bytes, byteorder='big', signed=False)

def blob_to_int(val):
    return int.from_bytes(val, byteorder='big', signed=False)

### SPLITTING INTEGERS

def split_integer(x, chunk_size=128): # CHANGED DEFAULT VALUE FROM 64 (!)
    mask = (1 << chunk_size) - 1
    chunks = []
    while x > 0:
        chunk = x & mask
        chunks.append(chunk)
        x >>= chunk_size
    return chunks

def merge_integer(lst, chunk_size=64):
    x = 0
    i = 0
    for chunk in lst:
        x |= (chunk << (chunk_size*i))
        i += 1
    return x

### PERMUTATIONS:

def sorting_perm(lst):
    sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1])
    permutation = tuple(index for index, _ in sorted_pairs)
    return permutation

def repeats(lst):
    tab = sorted(lst)
    counts = tuple(len(list(group)) for _, group in groupby(tab))
    return counts
    
@lru_cache(maxsize=None)
def perms(rptlist):
    permlist = [list(permutations(range(p))) for p in rptlist]
    for _ in range(len(rptlist)-1):
        m = max(permlist[-2][0])
        permlist[-1] = [[j+m+1 for j in s] for s in permlist[-1]]
        permlist[-2] = [list(p)+list(s) for p in permlist[-2] for s in permlist[-1]]
        permlist = permlist[:-1]
    return tuple(tuple(p) for p in permlist[0])

### DIFFERENT ENCODINGS OF COMPLEXES:

def zeroone(lst,tab):
    zo = 0
    for e in lst:
        zo |= (1 << tab.index(e))
    return zo

def fromzeroone(zo,tab):
    fromzo = []
    lst = [int(bit) for bit in bin(zo)[2:]]
    lst.reverse()
    lst += [0 for _ in range(len(tab)-len(lst))]
    for i in range(len(lst)):
        if lst[i] == 1:
            fromzo.append(tab[i])
    return(sorted(fromzo))

### FACES, BOUNDARIES, DEGREES:

@lru_cache(maxsize=None)
def cubes(n,k=None):
    prod = product(range(3),repeat=n)
    if k is not None:
        return tuple(c for c in prod if c.count(2) == k)
    else:
        return tuple(prod)

@lru_cache(maxsize=None)
def bdry(c):
    b = []
    for i in range(len(c)):
        if c[i] == 2:
            b.append(c[:i]+(0,)+c[i+1:])
            b.append(c[:i]+(1,)+c[i+1:])
    return tuple(b)

@lru_cache(maxsize=None)
def bbdry(c):
    return tuple(sorted(list(set(v for s in bdry(c) for v in bdry(s)))))

@lru_cache(maxsize=None)
def boundaries(n):
    return tuple(zeroone(bdry(c),cubes(n,1)) for c in cubes(n,2))

@lru_cache(maxsize=None)
def bboundaries(n):
    return tuple(zeroone(bbdry(c),cubes(n,0)) for c in cubes(n,2))

@lru_cache(maxsize=None)
def edgesquares(k):
    return zeroone([c for c in cubes(n,2) if cubes(n,1)[k] in bdry(c)],cubes(n,2))

### AUTOMORPHISMS AND CANONICAL LABELING:

@lru_cache(maxsize=None)
def edgeboundaries(n):
    return tuple(zeroone(bdry(c),cubes(n,0)) for c in cubes(n,1))

@lru_cache(maxsize=None)
def fake_octal(x):
    return int(bin(x)[2:],8)

@lru_cache(maxsize=None)
def nbhd(v):
    return tuple(v[:i]+(1-v[i],)+v[i+1:] for i in range(len(v)))

@lru_cache(maxsize=None)
def nnbhd(i):
    v = cubes(n,0)[i]
    return tuple(cubes(n,0).index(u) for u in nbhd(v))

@lru_cache(maxsize=None)
def permlist(nbdegs):
    pi = sorting_perm(nbdegs)
    return tuple(tuple(pi[i] for i in tau) for tau in perms(repeats(nbdegs)))

@lru_cache(maxsize=None)
def renamecube(c,p):
    newc = tuple(1-c[i] if p[0][i] == 1 and c[i] <= 1 else c[i] for i in range(len(c)))
    return tuple(newc[i] for i in p[1])

@lru_cache(maxsize=None)
def faceperm(p):
    squareindices = {c:index for index,c in enumerate(cubes(n,2))}
    return tuple(squareindices[renamecube(c,p)] for c in cubes(n,2))

### FUNCTIONS FOR TESTING LINKS:

@lru_cache(maxsize=None)
def twos(c):
    return tuple(max(0,s-1) for s in c)

def cycleq(tup):
    return tuple(sum(x) % 2 for x in zip(*tuple(twos(c) for c in tup))) == tuple(0 for _ in range(n))

@lru_cache(maxsize=None)
def cycles3(n,i):
    v = cubes(n,0)[i]
    lst = [s for s in cubes(n,2) if v in bbdry(s)]
    cycs = tuple(cyc for cyc in combinations(lst,3) if cycleq(cyc))
    return tuple(zeroone(cyc,cubes(n,2)) for cyc in cycs)

@lru_cache(maxsize=None)
def cycles4(n,i):
    v = cubes(n,0)[i]
    lst = [s for s in cubes(n,2) if v in bbdry(s)]
    cycs = tuple(cyc for cyc in combinations(lst,4) if cycleq(cyc))
    return tuple(zeroone(cyc,cubes(n,2)) for cyc in cycs)

@lru_cache(maxsize=None)
def cycles5(n,i):
    v = cubes(n,0)[i]
    lst = [s for s in cubes(n,2) if v in bbdry(s)]
    cycs = tuple(cyc for cyc in combinations(lst,5) if cycleq(cyc))
    return tuple(zeroone(cyc,cubes(n,2)) for cyc in cycs)

def initialize_rust_globals(): ### PRECOMPUTE VALUES AND PASS THEM TO RUST.
    def cache_enabled():
        v = os.environ.get("PRECOMPUTE_CACHE", "1").strip().lower()
        return v not in ("0", "false", "off", "no")

    def cache_path():
        cache_dir = os.environ.get("PRECOMPUTE_CACHE_DIR", ".")
        return os.path.join(cache_dir, f"precompute_cache_n{n}_v{PRECOMPUTE_CACHE_VERSION}.pkl")

    def timed_build(name, builder):
        t0 = time.time()
        value = builder()
        print(f"Precompute {name}: {time.time()-t0:.2f}s")
        return value

    def build_payload():
        return {
            "version": PRECOMPUTE_CACHE_VERSION,
            "n": n,
            "n2": n2,
            "n1": n1,
            "n0": n0,
            "boundaries": timed_build("boundaries", lambda: [split_integer(i) for i in boundaries(n)]),
            "bboundaries": timed_build("bboundaries", lambda: [split_integer(fake_octal(bboundaries(n)[i])) for i in range(n2)]), # ASSUMING n <= 7
            "edgeboundaries": timed_build("edgeboundaries", lambda: [split_integer(fake_octal(edgeboundaries(n)[i])) for i in range(n1)]), # ASSUMING n <= 7
            "nnbhd": timed_build("nnbhd", lambda: [list(nnbhd(i)) for i in range(n0)]),
            "cubes": timed_build("cubes", lambda: [list(c) for c in cubes(n,0)]),
            "edgesquares": timed_build("edgesquares", lambda: [split_integer(edgesquares(i)) for i in range(n1)]),
            "cycles3": timed_build("cycles3", lambda: [[split_integer(j) for j in cycles3(n,i)] for i in range(n0)]),
            "cycles4": timed_build("cycles4", lambda: [[split_integer(j) for j in cycles4(n,i)] for i in range(n0)]),
            "cycles5": timed_build("cycles5", lambda: [[split_integer(j) for j in cycles5(n,i)] for i in range(n0)] if n >= 7 else [[] for _ in range(n0)]),
        }

    payload = None
    cpath = cache_path()
    if cache_enabled() and os.path.exists(cpath):
        try:
            with open(cpath, "rb") as f:
                loaded = pickle.load(f)
            if (
                isinstance(loaded, dict)
                and loaded.get("version") == PRECOMPUTE_CACHE_VERSION
                and loaded.get("n") == n
                and loaded.get("n2") == n2
                and loaded.get("n1") == n1
                and loaded.get("n0") == n0
            ):
                payload = loaded
                print(f"Loaded precompute cache from {cpath}.")
        except Exception as e:
            print(f"WARNING: failed to load precompute cache ({e}); recomputing.")

    if payload is None:
        print("Building precomputed data ...")
        payload = build_payload()
        if cache_enabled():
            try:
                os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
                with open(cpath, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved precompute cache to {cpath}.")
            except Exception as e:
                print(f"WARNING: failed to save precompute cache ({e}).")

    if n == 7:
        print("WARNING!!! For n = 7, the precomputing step needs a significant amount of memory.")
    rust_code.load_precomputed_values(
        n,
        n2,
        n1,
        n0,
        chunksize,
        payload["boundaries"],
        payload["bboundaries"],
        payload["edgeboundaries"],
        payload["nnbhd"],
        payload["cubes"],
        payload["edgesquares"],
        payload["cycles3"],
        payload["cycles4"],
        payload["cycles5"]
    )

def extendonce(ncplx):
    return [merge_integer(cx) for cx in rust_code.extendonce(split_integer(ncplx))]

def disconnected_withbdry_extendonce(ncplx):
    return [merge_integer(cx) for cx in rust_code.disconnected_withbdry_extendonce(split_integer(ncplx))]

def withbdry_extendonce(ncplx):
    return [merge_integer(cx) for cx in rust_code.withbdry_extendonce(split_integer(ncplx))]

def disconnected_extendonce(ncplx):
    return [merge_integer(cx) for cx in rust_code.disconnected_extendonce(split_integer(ncplx))]

def cubicalcanlabel(ncplx):
    return merge_integer(rust_code.cubicalcanlabel(split_integer(ncplx)))

def testedges(ncplx):
    return rust_code.testedges(split_integer(ncplx))

def main_loop(imin,imax,ngood,initial_lencplxs,initial_lengoodcplxs,dbprefix,db_name):
    #return rust_code.test()
    return rust_code.main_loop(imin,imax,ngood,initial_lencplxs,initial_lengoodcplxs,dbprefix,db_name)
