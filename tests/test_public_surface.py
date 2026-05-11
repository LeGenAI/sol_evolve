from pathlib import Path


def test_long_form_public_directories_are_absent():
    root = Path(__file__).resolve().parents[1]

    for relpath in (
        "src/solevolve/" + "scripts",
        "src/solevolve/" + "experiments",
        "src/solevolve/" + "examples",
        "scripts",
        "experiments",
        "examples",
        "paper_" + "submission",
        "results/raw_outputs",
        "results/" + "tmp",
    ):
        assert not (root / relpath).exists()


def test_removed_private_paths_do_not_leak_into_public_files():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("/" + "Users/", "Code" + "Evolve", "results/" + "tmp")
    allowed = {root / "docs" / "paper_artifact.md"}

    for path in root.rglob("*"):
        if path.is_dir() or path.name == ".env":
            continue
        if any(part in path.parts for part in (".git", ".pytest_cache", ".venv", "__pycache__")):
            continue
        if any(part.endswith(".egg-info") for part in path.parts):
            continue
        if path in allowed or path.suffix in {".pyc", ".png", ".jpg", ".jpeg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            assert token not in text, f"{token} leaked in {path}"
