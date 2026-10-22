"""Run deterministic pre-submission checks for the LaTeX thesis project."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DASHES = {"\u2011": "non-breaking hyphen", "\u2013": "en dash", "\u2014": "em dash"}
INTENTIONAL_PLACEHOLDERS = (
    "20XX--20XX",
    "[Supervisor Name]",
    "[Co-supervisor Name]",
    "[Candidate Name]",
    "[Surname]",
    "[000000]",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_keys(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def check_hash(path: Path, expected: str, errors: list[str], label: str) -> None:
    if not path.is_file():
        errors.append(f"Missing {label}: {path}")
    elif sha256(path) != expected:
        errors.append(f"SHA-256 mismatch for {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    thesis = project / "thesis"
    errors: list[str] = []
    warnings: list[str] = []

    tex_files = [thesis / "thesis_main.tex", thesis / "titlepage.tex"]
    for folder in ("Chapters", "appendices", "tables", "generated"):
        tex_files.extend(sorted((thesis / folder).glob("*.tex")))
    tex_files = [path for path in tex_files if path.is_file()]
    tex_by_path = {path: path.read_text(encoding="utf-8") for path in tex_files}
    combined = "\n".join(tex_by_path.values())

    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        errors.append("Duplicate LaTeX labels: " + ", ".join(duplicate_labels))

    references: set[str] = set()
    for raw in re.findall(r"\\(?:[Cc]ref|ref|pageref|eqref)\{([^}]+)\}", combined):
        references.update(split_keys(raw))
    missing_labels = sorted(references - set(labels))
    if missing_labels:
        errors.append("Undefined LaTeX references: " + ", ".join(missing_labels))

    bib_path = thesis / "references.bib"
    bib_text = bib_path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))
    citations: set[str] = set()
    citation_pattern = r"\\(?:paren|text|auto|foot|smart|super)?cite\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}"
    for raw in re.findall(citation_pattern, combined):
        citations.update(split_keys(raw))
    missing_citations = sorted(citations - bib_keys)
    if missing_citations:
        errors.append("Citation keys missing from references.bib: " + ", ".join(missing_citations))
    uncited_entries = sorted(bib_keys - citations)
    if uncited_entries:
        warnings.append("Uncited bibliography entries: " + ", ".join(uncited_entries))

    entry_starts = list(re.finditer(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))
    bibliography_without_locator: list[str] = []
    for index, match in enumerate(entry_starts):
        end = entry_starts[index + 1].start() if index + 1 < len(entry_starts) else len(bib_text)
        entry = bib_text[match.start() : end]
        if not re.search(r"(?mi)^\s*(?:doi|url)\s*=", entry):
            bibliography_without_locator.append(match.group(1))
    if bibliography_without_locator:
        errors.append(
            "Bibliography entries without DOI or stable URL: "
            + ", ".join(bibliography_without_locator)
        )

    missing_graphics: list[str] = []
    for graphic in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", combined):
        candidate = thesis / "figures" / graphic
        if not candidate.is_file():
            missing_graphics.append(graphic)
    if missing_graphics:
        errors.append("Missing thesis graphics: " + ", ".join(sorted(set(missing_graphics))))

    for path, content in tex_by_path.items():
        for character, name in DASHES.items():
            if character in content:
                errors.append(f"Unicode {name} in {path.relative_to(project)}")

    main_text = (thesis / "thesis_main.tex").read_text(encoding="utf-8")
    placeholders = sorted({token for token in INTENTIONAL_PLACEHOLDERS if token in main_text})
    if placeholders:
        warnings.append("Administrative placeholders remain in the thesis metadata.")

    results_root = project / "results" / "verified"
    results_index_path = results_root / "results_index.json"
    results_index = load_json(results_index_path)
    for entry in results_index["sources"]:
        path = results_root / entry["relative_path"]
        check_hash(path, entry["sha256"], errors, f"indexed result {entry['relative_path']}")
    determinism = results_index["determinism_check"]
    check_hash(
        (results_root / determinism["relative_path"]).resolve(),
        determinism["sha256"],
        errors,
        "determinism check",
    )

    values_manifest_path = thesis / "generated" / "results_values_manifest.json"
    values_manifest = load_json(values_manifest_path)
    check_hash(
        Path(values_manifest["output"]),
        values_manifest["output_sha256"],
        errors,
        "generated result macros",
    )
    check_hash(
        results_index_path,
        values_manifest["results_index_sha256"],
        errors,
        "results index used by result macros",
    )
    for label, source in values_manifest["sources"].items():
        check_hash(Path(source["path"]), source["sha256"], errors, f"macro source {label}")

    assets_manifest = load_json(thesis / "generated" / "act8_assets_manifest.json")
    for output in assets_manifest["outputs"]:
        check_hash(Path(output["path"]), output["sha256"], errors, "thesis asset")
    for source in assets_manifest["verified_inputs"]:
        check_hash(Path(source["path"]), source["sha256"], errors, "verified asset input")
    for staged in assets_manifest["staged_sources"]:
        check_hash(Path(staged["source"]), staged["source_sha256"], errors, "staged figure source")
        check_hash(Path(staged["manifest"]), staged["manifest_sha256"], errors, "figure manifest")

    required_directories = (
        "thesis",
        "code",
        "reports",
        "results/verified",
        "scripts_figures",
        "data_notes",
        "environment",
        "provenance",
    )
    missing_directories = [name for name in required_directories if not (project / name).is_dir()]
    if missing_directories:
        errors.append("Missing repository directories: " + ", ".join(missing_directories))
    for name in ("README.md", "CHANGELOG.md", ".gitignore"):
        if not (project / name).is_file():
            errors.append(f"Missing repository file: {name}")

    result = {
        "schema_version": "1.0",
        "status": "pass" if not errors else "fail",
        "checks": {
            "tex_files_checked": len(tex_files),
            "labels": len(labels),
            "references": len(references),
            "bibliography_entries": len(bib_keys),
            "citations_used": len(citations),
            "graphics_referenced": len(
                re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", combined)
            ),
            "indexed_result_files": len(results_index["sources"]),
            "asset_outputs": len(assets_manifest["outputs"]),
            "intentional_placeholders": placeholders,
        },
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
