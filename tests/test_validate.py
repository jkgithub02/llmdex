"""The validate command (R7.5).

The vault is hand-edited, so the schema is only a contract if something checks
it. The exit code is the contract: zero means every document parsed.
"""

from api.validate import main

GOOD = "---\nmodel_id: a/one\n---\n"
NO_MODEL_ID = "---\nname: a document with no model_id\n---\n"


def _vault(tmp_path, **documents: str):
    models = tmp_path / "vault" / "models"
    models.mkdir(parents=True)
    for stem, text in documents.items():
        # newline="" or Windows writes \r\n and every document fails the
        # frontmatter regex, making the healthy-vault case fail too.
        (models / f"{stem}.md").write_text(text, encoding="utf-8", newline="")
    return tmp_path / "vault"


def test_validate_exits_zero_and_says_nothing_on_a_healthy_vault(tmp_path, monkeypatch, capsys):
    """Silence on success, so a CI job scraping stderr sees only real failures."""
    monkeypatch.setenv("LLMDEX_VAULT", str(_vault(tmp_path, good=GOOD)))

    code = main()

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_validate_exits_non_zero_and_names_every_broken_document(tmp_path, monkeypatch, capsys):
    vault = _vault(tmp_path, good=GOOD, broken_one=NO_MODEL_ID, broken_two=NO_MODEL_ID)
    monkeypatch.setenv("LLMDEX_VAULT", str(vault))

    code = main()

    err = capsys.readouterr().err
    assert code == 1
    assert "broken_one.md" in err
    assert "broken_two.md" in err
    assert "good.md" not in err
