import os
import time
import sqlite3 as s3
import builtins
import glob
import sys

import user_interface as ui
import functions as fn

# REDEFINE print FUNCTION TO FLUSH AUTOMATICALLY.
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

def remove_sqlite_artifacts(db_path):
    for suffix in ("", "-wal", "-shm"):
        path = db_path + suffix
        if os.path.exists(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

def remove_sqlite_glob(pattern):
    for path in glob.glob(pattern):
        if path.endswith(("-wal", "-shm")):
            continue
        remove_sqlite_artifacts(path)

def final_good_shard_pattern(dbprefix, n):
    return f"{dbprefix}cplxs{n}_good_part_*.db"

def legacy_final_good_shard_pattern(n):
    return f"cplxs{n}_good_part_*.db"

def legacy_good_shard_pattern():
    return "goodcplxs_s*.db"

def has_existing_final_good_shards(dbprefix, n):
    return (
        bool(glob.glob(final_good_shard_pattern(dbprefix, n)))
        or bool(glob.glob(legacy_final_good_shard_pattern(n)))
        or bool(glob.glob(legacy_good_shard_pattern()))
    )

def env_truthy(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "on", "yes", "y")

def main():
    ### SET PARAMETERS
    n, chunksize, dbprefix, surf_type = ui.user_interface()
    # chunksize = 10000
    # n = 6
    # print("OVERRIDE: chunksize = "+str(chunksize)+", n = "+str(n)+".")
    fn.initialize_globals(n,chunksize)
    fn.initialize_rust_globals()
    imin = 2 # default: 2
    imax = fn.n2 #fn.n2 # default: fn.n2
    cplxs = [1]
    #cplxs = [1<<i for i in range(fn.n2)] # For enumeration without isomorphism reduction.
    lencplxs = len(cplxs)
    db_name = dbprefix+f'cplxs{n}.db'
    existing_main_output = os.path.exists(db_name)
    existing_good_shards = has_existing_final_good_shards(dbprefix, n)
    if existing_main_output or existing_good_shards:
        if env_truthy("AUTO_OVERWRITE_DB", default=False):
            response = "o"
        else:
            response = input(
                f"WARNING!!! Existing output detected: {db_name}"
                + (" and final good shard files" if existing_good_shards else "")
                + ".\nOverwrite [O] or quit [Q]? "
            ).lower()
        if response == 'o':
            print("Overwriting previous output.")
            remove_sqlite_artifacts(db_name)
            remove_sqlite_glob(final_good_shard_pattern(dbprefix, n))
            remove_sqlite_glob(legacy_final_good_shard_pattern(n))
            remove_sqlite_glob(legacy_good_shard_pattern())
        else:
            print("Cannot proceed without permission. Exiting computation.")
            exit()

    ### CREATE AND INITIALIZE DATABASE (OR ATTEMPT TO CONTINUE PREVIOUS COMPUTATION)
    t0 = time.time()

    conn = s3.connect(db_name)
    cursor = conn.cursor()
    remove_sqlite_glob('tempcplxs*.db')
    remove_sqlite_glob('templabels*.db')
    remove_sqlite_glob('tempgood*.db')
    cplxconn = s3.connect('tempcplxs1.db')
    cplxcursor = cplxconn.cursor()
    
    if imin == 2:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goodcplxs (
                cplx BLOB PRIMARY KEY,
                id INTEGER
            )
        """)
        if dbprefix == "b" or dbprefix == "db":
            cursor.execute("INSERT INTO goodcplxs (cplx, id) VALUES (?, ?)", 
                        (fn.int_to_blob(cplxs[0]), 1))
        conn.commit()
        cplxcursor.execute("""
            CREATE TABLE IF NOT EXISTS cplxs1 (
                id INTEGER PRIMARY KEY,
                cplx BLOB
            )
        """)
        for val1 in cplxs:
            cplxcursor.execute("INSERT INTO cplxs1 (cplx) VALUES (?)", 
                        (fn.int_to_blob(val1),))
        cplxconn.commit()
    else:
        print("FAILED: imin != 2.")
        sys.exit(1)

    print(f"Enumerating {surf_type}; n = {n}, chunk size = {chunksize}.\nStoring data to database {db_name}.")

    write_to_text = False
    if write_to_text:
        print("Also storing final result as a .txt file.")

    cursor.execute(f"SELECT COUNT(*) FROM goodcplxs")
    ngood = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    cplxcursor.close()
    cplxconn.close()
    
    ### MAIN LOOP OF THE COMPUTATION (STARTING FROM POINT OF INTERRUPTION IF NEEDED)
    fn.main_loop(imin,imax,ngood,lencplxs,ngood,dbprefix,db_name)
    if write_to_text:
        if env_truthy("GOOD_SHARDING", default=False) and not env_truthy("GOOD_FINAL_MERGE", default=False):
            print("Text export currently requires merged good output. Re-run with GOOD_FINAL_MERGE=1 or add a shard-aware exporter.")
            sys.exit(1)
        conn = s3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM goodcplxs ORDER BY id")
        writecplxs = cursor.fetchall()
        writestring = "".join([str(fn.fromzeroone(fn.blob_to_int(c[0]),fn.cubes(n,2)))+"\n" for c in writecplxs])
        g = open(db_name[:-3]+"_result.txt","w")
        g.write(writestring)
        g.close()
        cursor.close()
        conn.close()

    print("Total time of computation: "+str(time.time()-t0))

if __name__ == "__main__":
    main()
