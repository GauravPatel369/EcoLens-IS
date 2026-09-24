"""
Static sanity check for the generated dashboard HTML.

WHY THIS EXISTS
---------------
On 16 Sep the heat-map panel was added with `renderRiskMap()` defined inside
`showComparison()` but called from `doSearch()` -- a different scope. JavaScript raised
ReferenceError, `doSearch()` aborted at that line, and the entire results list disappeared
along with the map.

It was "verified" by grepping the HTML for the function name and the data constant. Both
were present. Presence proves the text was written; it proves nothing about whether the
page runs. This checks the things that actually break:

  1. every function called at top level is DEFINED at top level
  2. every data constant the renderers read is present and non-empty
  3. braces balance inside the <script> block
  4. the panel ids the JS reaches for exist in the markup

Run:
    python tools_check_dashboard.py [path-to-html]
"""

import json
import re
import sys

DEFAULT = "outputs/dashboards/retrieval_dashboard.html"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    src = open(path, encoding="utf-8", errors="ignore").read()
    fails = []

    # the big inline script is the last <script> without a src attribute
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", src, re.S)
    js = max(blocks, key=len) if blocks else ""
    print(f"  inline script: {len(js):,} chars")

    top = set(re.findall(r"^function ([A-Za-z_$][\w$]*)\s*\(", js, re.M))
    nested = set(re.findall(r"^\s+function ([A-Za-z_$][\w$]*)\s*\(", js, re.M))
    print(f"  top-level functions: {len(top)}   nested: {len(nested)}")

    # 1. scope: a nested-only function must never be called from column-0 code
    top_level_code = "\n".join(l for l in js.split("\n")
                               if l and not l.startswith((" ", "\t")))
    for fn in sorted(nested - top):
        if re.search(r"\b" + re.escape(fn) + r"\s*\(", top_level_code):
            fails.append(f"SCOPE: {fn}() is nested but called from top-level code")

    # 2. data constants present and non-trivial
    for const in ["ALL_MODEL_DATA", "EXPLAIN_DATA", "FORECASTS", "RISKGRID"]:
        m = re.search(r"const " + const + r"\s*=\s*(\{.*?\});\s*\n", js, re.S)
        if not m:
            fails.append(f"DATA: const {const} missing")
            continue
        try:
            obj = json.loads(m.group(1))
        except Exception as e:
            fails.append(f"DATA: {const} is not valid JSON ({type(e).__name__})")
            continue
        print(f"  {const:<15} {len(obj)} top-level keys")
        if not obj:
            fails.append(f"DATA: {const} is empty")

    # 3. braces balance
    if js.count("{") != js.count("}"):
        fails.append(f"BRACES: {js.count('{')} open vs {js.count('}')} close")

    # 4. every getElementById target exists in the markup
    for eid in sorted(set(re.findall(r"getElementById\('([^']+)'\)", js))):
        if f'id="{eid}"' not in src:
            fails.append(f"DOM: getElementById('{eid}') has no matching element")

    print()
    if fails:
        print(f"  {len(fails)} PROBLEM(S):")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
