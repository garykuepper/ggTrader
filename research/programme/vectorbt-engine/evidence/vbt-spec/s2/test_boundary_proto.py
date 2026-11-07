"""Prototype: AST boundary test -- vectorbt may only be touched inside ggTrader/engine/."""
import ast, pathlib, sys
ROOT = pathlib.Path(sys.argv[1])
ALLOWED = ROOT / "src/ggTrader/engine"
BANNED_MODS = ("vectorbt", "vectorbt_rust")
def violations(path):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in BANNED_MODS: yield n.lineno, f"import {a.name}"
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in BANNED_MODS:
            yield n.lineno, f"from {n.module} import ..."
        elif isinstance(n, ast.Attribute) and n.attr == "vbt":
            yield n.lineno, ".vbt accessor"
        elif isinstance(n, ast.Constant) and n.value in BANNED_MODS:
            yield n.lineno, f"string {n.value!r} (dynamic import?)"
tot = {}
for p in list((ROOT/"src").rglob("*.py")) + list((ROOT/"scripts").rglob("*.py")) + list((ROOT/"tests").rglob("*.py")) + [ROOT/"ggt.py"]:
    if ALLOWED in p.parents: continue
    v = list(violations(p))
    if v: tot[str(p.relative_to(ROOT))] = v
for k, v in tot.items(): print(k, [x[1] for x in v][:4], len(v))
print("files:", len(tot))
