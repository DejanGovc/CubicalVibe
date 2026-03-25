import os

def user_interface():
    def env_truthy(name, default=False):
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() in ("1", "true", "on", "yes", "y")

    def ask_choice(prompt, allowed):
        answer = input(prompt).strip().lower()
        if answer not in allowed:
            print("Invalid input. Exiting computation.")
            exit()
        return answer

    def ask_dimension():
        while True:
            try:
                val = int(input("Please enter the dimension of the cube I^n: "))
            except ValueError:
                continue
            if 2 < val < 8:
                return val
            print("The dimension must be at least 3 and at most 7.")

    print("\033[H\033[J", end="")
    print("   +----+")
    print("  /    /|")
    print(" +----+ |  +--------------------------------------------+")
    print(" |    | +   \\ CubicalPy: enumeration of cubical surfaces \\")
    print(" |    |/     +--------------------------------------------+")
    print(" +----+\n")

    default_chunksize = 1_000_000
    default_n = 6
    default_surf_type = "closed surfaces"
    default_dbprefix = ""
    chunksize = default_chunksize

    print(
        f"By default, computing (connected) {default_surf_type} "
        f"in the {default_n}-cube."
    )

    if env_truthy("AUTO_UI_CONTINUE", default=False):
        response1 = "c"
    else:
        response1 = ask_choice("Continue [C] or choose your own parameters [P]? ", {"c", "p"})
    if response1 == "c":
        n = default_n
        surf_type = default_surf_type
        dbprefix = default_dbprefix
    else:
        response2 = ask_choice("Closed surfaces [C] or surfaces with boundary [B]? ", {"c", "b"})
        if response2 == "b":
            surf_type = "surfaces with boundary"
            dbprefix = "b"
        else:
            surf_type = "closed surfaces"
            dbprefix = ""

        response3 = ask_choice("Include disconnected surfaces? [Y/N] ", {"y", "n"})
        if response3 == "y":
            surf_type += " (including disconnected)"
            dbprefix = "db" if dbprefix == "b" else "d"

        print(f"Using default chunksize ({default_chunksize}).")
        n = ask_dimension()

    check_links_env = os.environ.get("CHECK_VERTEX_LINKS", "1").strip().lower()
    check_links_enabled = check_links_env not in ("0", "false", "off", "no")
    if not check_links_enabled:
        print("WARNING!!! Vertex-link checks are disabled (CHECK_VERTEX_LINKS=0); singular surfaces may appear.")
    elif dbprefix in ("b", "db"):
        if n >= 4:
            print("WARNING!!! Full link connectedness is not guaranteed in boundary/disconnected modes; singular surfaces may appear.")
    elif n >= 7:
        # Connected closed surfaces in n=6: current link tests rule out disconnected S^1-links.
        print("WARNING!!! For n >= 7, full link connectedness is not guaranteed by current local tests; singular surfaces may appear.")

    return n, chunksize, dbprefix, surf_type
