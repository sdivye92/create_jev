import json, sys, os, re
nb_path = sys.argv[1]
nb = json.load(open(nb_path))
src = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
src = "\n".join(l for l in src.split("\n") if not l.startswith("%"))
assert "SMOKE = False" in src
src = src.replace("SMOKE = False", "SMOKE = True", 1)
os.chdir(os.path.dirname(os.path.abspath(nb_path)))
exec(compile(src, nb_path, "exec"), {"__name__": "__main__"})
