"""Snapshot -> document (R1.1, R1.4, R1.5, R4.1).

Ingest does exactly two things: derive what the structured files support, and
hand the result to the store to be merged. It writes nothing it cannot defend,
and it never touches ``manual``, ``measured`` or ``extracted`` -- those belong to
a person, and quietly overwriting them is the failure this design exists to avoid.

Extraction is deliberately absent. Prose fields are entered by hand for now, so
there is no path here by which a language model can invent a value.
"""

from datetime import UTC, datetime

from app.core.document import DEFAULT_CONTEXT, Checkpoint, ModelDoc
from app.core.store import Store
from app.features.models.derive import derive, group_gguf_files
from app.features.models.fetch import RepoSnapshot


def _vendor_of(model_id: str) -> str | None:
    return model_id.split("/")[0] if "/" in model_id else None


def _display_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-", " ")


def build_document(
    snapshot: RepoSnapshot,
    context: int = DEFAULT_CONTEXT,
    today: str | None = None,
) -> ModelDoc:
    """Turn a snapshot into a document. Pure: no store, no network, no clock unless asked."""
    ingested = today or datetime.now(tz=UTC).date().isoformat()
    config = snapshot.config or {}

    grouped = group_gguf_files(snapshot.siblings)
    if len(grouped) > 1:
        # R4.1 - a GGUF shelf is several artifacts, so it becomes several checkpoints.
        checkpoints = [
            Checkpoint(
                repo=snapshot.model_id,
                quantization=quant,
                card_revision=snapshot.revision,
                ingested=ingested,
                derived=derive(
                    config,
                    siblings=grouped[quant],
                    safetensors_total=snapshot.safetensors_total,
                    context=context,
                ),
            )
            for quant in sorted(grouped)
        ]
    else:
        checkpoints = [
            Checkpoint(
                repo=snapshot.model_id,
                quantization=next(iter(grouped), None),
                card_revision=snapshot.revision,
                ingested=ingested,
                derived=derive(
                    config,
                    siblings=snapshot.siblings,
                    safetensors_total=snapshot.safetensors_total,
                    # Not passed to the GGUF branch above: those headers describe
                    # the safetensors artifact, and a GGUF checkpoint is a
                    # different set of files.
                    tensor_headers=snapshot.tensor_headers,
                    context=context,
                ),
            )
        ]

    return ModelDoc(
        model_id=snapshot.model_id,
        name=_display_name(snapshot.model_id),
        vendor=_vendor_of(snapshot.model_id),
        checkpoints=checkpoints,
    )


def ingest(
    snapshot: RepoSnapshot,
    store: Store,
    context: int = DEFAULT_CONTEXT,
    today: str | None = None,
) -> ModelDoc:
    """Build and merge. Merging is what keeps a human's corrections alive (R6.5)."""
    doc = build_document(snapshot, context=context, today=today)
    return store.merge_ingest(doc, operation="ingest")
