"""Build the active LEFT manuscript with Pandoc MathML and headless Chromium."""
import argparse
from pathlib import Path
import subprocess
import tempfile

MANUSCRIPT = Path("manuscript/next_generation_20261115/draft.md")
FILTER = r"""
function Link(link)
  local prefix = "https://github.com/codexpker/Hybrid/blob/codex/latest/"
  local destination = link.target
  local shared = destination:match("^%.%./%.%./(.+)$")
  if shared then link.target = prefix .. shared
  elseif destination:match("^figures/") or destination:match("^REPRODUCIBILITY%.md") then
    link.target = prefix .. "manuscript/next_generation_20261115/" .. destination
  end
  return link
end
"""
CSS = """
<style>
@page { size: A4; margin: 16mm 17mm 19mm; @bottom-center { content: counter(page); font: 9pt sans-serif; color: #526174; } }
:root { color: #182333; }
body { max-width: 176mm; margin: 20px auto; padding: 0; font: 11pt/1.42 Georgia, serif; }
h1, h2, h3 { line-height: 1.2; color: #132c40; break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 0.36em; padding-bottom: 0.42em; border-bottom: 1px solid #9babba; }
h2 { font-size: 16pt; margin: 1.05em 0 0.42em; }
h3 { font-size: 12pt; margin: 0.9em 0 0.4em; }
p, ol, ul { margin: 0.5em 0; orphans: 3; widows: 3; }
a { color: #174969; text-decoration: none; }
math[display="block"] { margin: 0.55em 0; }
table { width: 100%; border-collapse: collapse; margin: 0.65em 0 0.8em; font: 8.5pt/1.3 "Segoe UI", sans-serif; }
thead { break-after: avoid; display: table-header-group; color: #17293a; background: #e8eff4; }
thead th { white-space: nowrap; }
td, th { text-align: left; vertical-align: top; padding: 3px 5px; border-bottom: 1px solid #d7e0e6; }
tr { break-inside: avoid; }
figure { margin: 0.7em 0 1.1em; break-inside: avoid; text-align: center; }
figure img { width: 100%; height: auto; max-height: 150mm; object-fit: contain; }
figcaption { text-align: left; margin: 0.3em 0; font: 9pt/1.35 "Segoe UI",sans-serif; color: #314254; }
code { font-size: 0.86em; overflow-wrap: anywhere; }
@media print {
  body { margin: 0; print-color-adjust: exact; }
  p:has(> strong:first-child):has(+ table) { break-after: avoid; }
}
</style>
"""


def render(repository, chromium, pdf, html):
    source = repository/MANUSCRIPT
    pdf.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="left-manuscript-") as temp:
        staging = Path(temp)
        filter_path, header = staging/"links.lua", staging/"style.html"
        filter_path.write_text(FILTER,encoding="utf-8")
        header.write_text(CSS,encoding="utf-8")
        final_html = html or staging/"manuscript.html"
        final_html.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["pandoc", str(source), "--from=markdown+implicit_figures",
                        "--to=html5", "--standalone", "--embed-resources", "--mathml",
                        "--metadata=pagetitle:Controlled empirical study of cross-domain LEFT",
                        "--metadata=lang:en",
                        f"--lua-filter={filter_path}", f"--include-in-header={header}",
                        f"--resource-path={source.parent}", "-o", str(final_html)],
                       cwd=source.parent,check=True)
        subprocess.run([str(chromium), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={staging/'chrome-profile'}", f"--print-to-pdf={pdf}",
                        final_html.as_uri()],check=True,timeout=60)
        assert pdf.stat().st_size > 10000


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("chromium", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--html", type=Path)
    a=parser.parse_args()
    render(a.repository.resolve(),a.chromium.resolve(),a.pdf.resolve(),a.html.resolve() if a.html else None)


if __name__=="__main__":
    main()
