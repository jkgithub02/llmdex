"""Ingest failures, named rather than lumped together (R1.6).

These live outside the models feature because ``core.http`` maps them to status
codes and every feature's router surfaces them. A feature must not have to
import another feature to raise the right error.

"gated", "private" and "does not exist" call for three different actions from
whoever ran the ingest, and telling them only that it "failed" wastes their
time.
"""


class IngestError(RuntimeError):
    """Base for every ingest failure that is the repository's fault, not ours."""


class RepoNotFound(IngestError):
    """No such repository on the Hub."""


class GatedRepo(IngestError):
    """The repository exists but requires accepting terms."""


class PrivateRepo(IngestError):
    """The repository exists but is not visible with these credentials."""


class AccessUndetermined(IngestError):
    """Hugging Face refused to say whether the repository exists.

    Anonymous requests for an unknown repo get ``401 Invalid username or
    password`` -- the same answer a private repo gives, deliberately, so that the
    Hub does not leak which private repositories exist. R1.6 asks us to name the
    failure; the honest name here is that we cannot tell yet, and the fix is a
    token.
    """
