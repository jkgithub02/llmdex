"""The store: markdown documents with YAML frontmatter, in a git repository.

Design constraints that shape everything here:

- **R7.6** a document must remain valid and readable if this application is
  deleted, so the file is plain text and the frontmatter is ordinary YAML.
- **R1.5** a write is atomic: the document is rendered in full before anything
  touches the disk, then written to a temporary file and renamed.
- **R1.4 / R6.5** ingest owns some blocks and must not touch others. That split
  lives in :meth:`Store.merge_ingest` and is the reason ingest never calls
  :meth:`Store.write` directly with a freshly-built document.
- **R4.7** every write that changes content lands as a git commit.
"""

import hashlib
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from app.core.schemas import (
    Benchmark,
    Checkpoint,
    Extracted,
    ExtractedBenchmarks,
    ModelDoc,
    Summary,
)

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

# Serialises the read-modify-write in merge_summary / merge_extraction /
# merge_ingest. The about and prose agents now call these concurrently on the
# same document (backend/agents/runner.py) -- without this, both threads can
# read before either writes, and the later write silently discards the
# earlier agent's block. A lock rather than the optimistic `expect_unchanged`
# retry: this is one process, the critical section is short, and a lock
# cannot livelock the way a retry loop under contention can.
#
# ponytail: one process-wide lock, so an unrelated model's write waits on this
# one's too. Fine at single-user scale; move to per-document locks, or the
# optimistic expect_unchanged retry already on Store.write, if this ever
# serves multiple processes.
_LOCK = threading.Lock()

# Blocks ingest is allowed to overwrite on an existing checkpoint. Anything not
# listed here belongs to a human and survives re-ingest (R6.5, R4.2).
INGEST_OWNED = ("card_revision", "ingested", "derived")
"""Blocks ingest may overwrite. ``extracted`` is deliberately absent: it belongs
to the extractor, which runs as its own operation, and listing it here meant every
re-ingest reset it to empty."""


class DocumentConflict(RuntimeError):
    """The document on disk changed since it was read."""


def slug_for(model_id: str) -> str:
    """``vendor/Model-Name`` -> ``vendor--model-name``, safe on every filesystem."""
    return model_id.replace("/", "--").lower()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkpoint_key(checkpoint: Checkpoint) -> tuple[str, str]:
    """What makes a checkpoint distinct: its repo and which artifact within it."""
    return (checkpoint.repo, checkpoint.quantization or "")


def _read(path: Path) -> str:
    """Read without newline translation, matching how :meth:`Store._atomic_write` writes."""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


class Store:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.benchmarks_dir = self.root / "benchmarks"

    # -- paths ------------------------------------------------------------

    def path_for(self, model_id: str) -> Path:
        return self.models_dir / f"{slug_for(model_id)}.md"

    def benchmark_path_for(self, slug: str) -> Path:
        return self.benchmarks_dir / f"{slug}.md"

    # -- render / parse ---------------------------------------------------

    def render(self, doc: ModelDoc) -> str:
        """Deterministic markdown+YAML. Same input, same bytes (R1.4).

        Nulls are written out rather than dropped: an absent field is a fact about
        the model card, and a reader of the raw file deserves to see it (R6.3).
        """
        payload = doc.model_dump(mode="json")
        frontmatter = yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100
        )
        body = f"# {doc.name or doc.model_id}\n\n`{doc.model_id}`\n"
        return f"---\n{frontmatter}---\n\n{body}"

    def parse(self, text: str) -> ModelDoc:
        match = FRONTMATTER.match(text)
        if not match:
            raise ValueError("document has no YAML frontmatter")
        doc = ModelDoc.model_validate(yaml.safe_load(match.group(1)) or {})
        doc._source_digest = _digest(text)
        return doc

    # -- read -------------------------------------------------------------

    def read(self, model_id: str) -> ModelDoc | None:
        path = self.path_for(model_id)
        if not path.exists():
            return None
        return self.parse(_read(path))

    def list_models(self) -> list[ModelDoc]:
        if not self.models_dir.exists():
            return []
        return [self.parse(_read(p)) for p in sorted(self.models_dir.glob("*.md"))]

    def read_benchmark(self, slug: str) -> Benchmark | None:
        path = self.benchmark_path_for(slug)
        if not path.exists():
            return None
        match = FRONTMATTER.match(_read(path))
        if not match:
            raise ValueError(f"benchmark {slug} has no YAML frontmatter")
        data = yaml.safe_load(match.group(1)) or {}
        data["explanation"] = match.group(2).strip()
        return Benchmark.model_validate(data)

    def list_benchmarks(self) -> list[Benchmark]:
        if not self.benchmarks_dir.exists():
            return []
        slugs = sorted(p.stem for p in self.benchmarks_dir.glob("*.md"))
        return [b for slug in slugs if (b := self.read_benchmark(slug)) is not None]

    def validate_all(self) -> list[tuple[Path, str]]:
        """Every document in the vault, parsed through the schema (R7.5).

        This walks the files itself rather than calling :meth:`list_models`, which
        raises on the first bad document. A validator that reports one failure out
        of nine is not a validator.

        ``Path.glob`` on a directory that does not exist yields nothing, so an
        empty vault is a pass rather than an error.
        """
        failures: list[tuple[Path, str]] = []
        for path in sorted(self.models_dir.glob("*.md")):
            try:
                self.parse(_read(path))
            except (ValueError, yaml.YAMLError) as exc:
                failures.append((path, str(exc)))
        for path in sorted(self.benchmarks_dir.glob("*.md")):
            try:
                self.read_benchmark(path.stem)
            except (ValueError, yaml.YAMLError) as exc:
                failures.append((path, str(exc)))
        return failures

    # -- write ------------------------------------------------------------

    def write(self, doc: ModelDoc, operation: str, expect_unchanged: bool = False) -> bool:
        """Render, write atomically, commit. Returns whether anything changed.

        Rendering happens before any filesystem mutation so that a failure to
        render leaves no partial document behind (R1.5).
        """
        text = self.render(doc)
        path = self.path_for(doc.model_id)

        if path.exists():
            current = _read(path)
            if expect_unchanged and doc._source_digest not in (None, _digest(current)):
                raise DocumentConflict(
                    f"{path.name} changed on disk since it was read; re-read and merge"
                )
            if current == text:
                return False  # nothing to commit (R4.7 says commit writes, not no-ops)

        self._atomic_write(path, text)
        doc._source_digest = _digest(text)
        self._commit(path, f"{operation}: {doc.model_id}")
        return True

    def write_benchmark(self, bench: Benchmark, operation: str) -> bool:
        payload = bench.model_dump(mode="json", exclude={"explanation"})
        frontmatter = yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100
        )
        text = f"---\n{frontmatter}---\n\n{bench.explanation}\n"
        path = self.benchmark_path_for(bench.slug)

        if path.exists() and _read(path) == text:
            return False
        self._atomic_write(path, text)
        self._commit(path, f"{operation}: benchmark {bench.slug}")
        return True

    def merge_ingest(self, incoming: ModelDoc, operation: str = "ingest") -> ModelDoc:
        """Fold a freshly-ingested document into whatever is already stored.

        Ingest refreshes only the blocks it owns. A hand-corrected field or a
        measured entry is not its business, and re-ingesting must not quietly
        discard the work of the person who put it there (R6.5, R4.2).
        """
        with _LOCK:
            existing = self.read(incoming.model_id)
            if existing is None:
                self.write(incoming, operation=operation)
                return incoming

            # Keyed on repo AND quantization: a GGUF repository publishes several
            # checkpoints under one repo name, so keying on repo alone silently
            # collapses them and leaves all but one holding a stale derived block.
            by_key = {_checkpoint_key(c): c for c in existing.checkpoints}
            for fresh in incoming.checkpoints:
                kept = by_key.get(_checkpoint_key(fresh))
                if kept is None:
                    existing.checkpoints.append(fresh)
                    continue
                for field in INGEST_OWNED:
                    setattr(kept, field, getattr(fresh, field))

            for field in ("name", "vendor", "released"):
                if (value := getattr(incoming, field)) is not None:
                    setattr(existing, field, value)

            existing.checkpoints.sort(key=_checkpoint_key)
            self.write(existing, operation=operation)
            return existing

    def merge_extraction(
        self,
        model_id: str,
        repo: str,
        quantization: str | None,
        extracted: Extracted,
        operation: str = "extract",
    ) -> ModelDoc:
        """Write one checkpoint's ``extracted`` block and nothing else (R3.x)."""
        return self._merge_block(model_id, repo, quantization, "extracted", extracted, operation)

    def merge_benchmarks(
        self,
        model_id: str,
        repo: str,
        quantization: str | None,
        benchmarks: ExtractedBenchmarks,
        operation: str = "benchmarks",
    ) -> ModelDoc:
        """Write one checkpoint's ``extracted_benchmarks`` block and nothing else.

        Its own block, so a re-run of the extractor cannot take the table with
        it: the two agents read the card at different times and each stamps its
        spans with the revision it actually read (R1.3, R3.3).
        """
        return self._merge_block(
            model_id, repo, quantization, "extracted_benchmarks", benchmarks, operation
        )

    def _merge_block(
        self,
        model_id: str,
        repo: str,
        quantization: str | None,
        field: str,
        value: object,
        operation: str,
    ) -> ModelDoc:
        """Replace one field of one checkpoint, leaving every other block alone.

        Neither agent ever creates a document or a checkpoint. If the target is
        not already in the store that is an error, not an invitation to invent
        one: a span is only meaningful against a card ingest has already read
        and recorded a revision for.
        """
        with _LOCK:
            doc = self.read(model_id)
            if doc is None:
                raise KeyError(f"{model_id} is not in the store")

            wanted = (repo, quantization or "")
            for checkpoint in doc.checkpoints:
                if _checkpoint_key(checkpoint) == wanted:
                    setattr(checkpoint, field, value)
                    break
            else:
                raise KeyError(
                    f"{model_id} has no checkpoint {repo!r} ({quantization or 'default'})"
                )

            self.write(doc, operation=operation)
            return doc

    def merge_summary(
        self,
        model_id: str,
        summary: Summary,
        operation: str = "summarise",
    ) -> ModelDoc:
        """Replace the model's ``summary`` block and nothing else.

        Like extraction, this never creates a document: a summary describes a
        model somebody has already ingested. Unlike extraction it has no
        checkpoint to find -- the block belongs to the model.

        Replacing rather than appending is deliberate. Keeping every past
        summary would make the page a changelog of an endpoint's opinions;
        ``generated_on`` says when the current one was written, which is the
        question a reader actually has.
        """
        with _LOCK:
            doc = self.read(model_id)
            if doc is None:
                raise KeyError(f"{model_id} is not in the store")

            doc.summary = summary
            self.write(doc, operation=operation)
            return doc

    def delete(self, model_id: str, operation: str = "delete") -> bool:
        """Remove a model's document. Returns False if there was nothing to remove.

        The only write in this class that destroys rather than adds, which is
        precisely why the vault is a git repository (R4.7): the removal lands as
        a commit, so the document and its whole history stay recoverable from the
        vault itself. That makes an archive flag unnecessary -- git already is
        the archive, and a flag would leave documents nobody wants sitting in
        every listing and validation pass.

        A missing file is not an error here. A delete that finds nothing has
        nothing to undo; the router turns that into a 404 because a caller naming
        a model that is not there is worth telling.
        """
        path = self.path_for(model_id)
        if not path.exists():
            return False

        path.unlink()
        self._commit(path, f"{operation}: {model_id}")
        return True

    def stub_benchmark(self, slug: str, referring_model: str) -> Benchmark:
        """R5.3 - a benchmark we have a score for but no document.

        The stub carries the slug and who referred to it, and is marked unwritten.
        The explanation stays empty because R5.2 forbids generating one.
        """
        existing = self.read_benchmark(slug)
        if existing is not None:
            if referring_model not in existing.referring_models:
                existing.referring_models = sorted({*existing.referring_models, referring_model})
                self.write_benchmark(existing, operation="record reference")
            return existing

        stub = Benchmark(slug=slug, unwritten=True, referring_models=[referring_model])
        self.write_benchmark(stub, operation="stub")
        return stub

    # -- internals --------------------------------------------------------

    def _atomic_write(self, path: Path, text: str) -> None:
        """Write via a temp file and rename, so a crash cannot leave half a document.

        ``newline=""`` matters more than it looks: the default would translate every
        ``\\n`` to CRLF on Windows, so the same document would render differently on
        different machines and R1.4's byte-identity claim would hold only per-platform.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def _commit(self, path: Path, message: str) -> None:
        if not (self.root / ".git").exists():
            return  # a vault without git still works; R4.7 just cannot apply
        rel = path.relative_to(self.root).as_posix()
        subprocess.run(["git", "add", "--", rel], cwd=self.root, check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root, check=False)
        if staged.returncode == 0:
            return  # nothing actually staged
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=self.root, check=True)


def new_checkpoint(repo: str, **kw: Any) -> Checkpoint:
    return Checkpoint(repo=repo, **kw)
