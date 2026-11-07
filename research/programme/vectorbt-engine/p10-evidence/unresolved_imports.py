"""List ggTrader imports (and ggTrader.* strings) in kept code that name a module that does not exist."""
import ast, sys
from pathlib import Path
root = Path(sys.argv[1])
mods = set()
for p in (root/"src").rglob("*.py"):
    parts = list(p.relative_to(root/"src").with_suffix("").parts)
    if parts[-1] == "__init__": parts = parts[:-1]
    for i in range(1, len(parts)+1): mods.add(".".join(parts[:i]))
bad = []
for folder in ("src","scripts","tests"):
    for p in (root/folder).rglob("*.py"):
        if "__pycache__" in p.parts: continue
        try: tree = ast.parse(p.read_text())
        except SyntaxError: continue
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import): names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                names = [n.module]
                names += [f"{n.module}.{a.name}" for a in n.names if f"{n.module}.{a.name}" in mods or True]
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("ggTrader."):
                names = [n.value]
            for name in names:
                if not name.startswith("ggTrader"): continue
                # resolvable if the longest module prefix exists and the remainder is <=1 attribute for from-imports
                parts = name.split(".")
                pref = max((i for i in range(1, len(parts)+1) if ".".join(parts[:i]) in mods), default=0)
                if pref == 0 or (pref < len(parts)-1 and not isinstance(n, ast.Constant)) and False:
                    bad.append((str(p.relative_to(root)), n.lineno, name))
                elif pref < len(parts) - 1:
                    bad.append((str(p.relative_to(root)), n.lineno, name))
for b in sorted(set(bad)): print(b)
print(len(set(bad)), "suspect")
