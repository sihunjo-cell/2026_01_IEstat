"""Convert notebooks -> code-only .py (strip markdown + all outputs).
Preserves code cell order exactly == original execution order.
Run from repo root. Originals (.ipynb) are left untouched.
"""
import json, os, glob

def convert(path):
    nb = json.load(open(path, encoding="utf-8"))
    out = ["# Auto-generated from %s (code cells only, markdown/outputs stripped)\n"
           % os.path.basename(path)]
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        # comment out IPython line magics / shell escapes so the .py stays valid
        lines = []
        for ln in src.split("\n"):
            s = ln.lstrip()
            if s.startswith("!") or s.startswith("%"):
                lines.append("# [nb-magic] " + ln)
            else:
                lines.append(ln)
        out.append("\n".join(lines).rstrip() + "\n")
    target = os.path.splitext(path)[0] + ".py"
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return target, os.path.getsize(path), os.path.getsize(target)

if __name__ == "__main__":
    rows = []
    for nb in sorted(glob.glob("*.ipynb")):
        t, before, after = convert(nb)
        rows.append((nb, before, after))
        print(f"{nb:34s} {before/1024:8.0f}KB -> {after/1024:6.1f}KB  ({t})")
    tb = sum(r[1] for r in rows); ta = sum(r[2] for r in rows)
    print(f"\nTOTAL ipynb {tb/1024/1024:.1f}MB  ->  py {ta/1024:.0f}KB")
