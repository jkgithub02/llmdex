"""Domain exceptions every feature's service layer may raise, named rather than
lumped together (R1.6).

These live outside any one feature because ``core.http`` maps the ingest
family to status codes and every feature's router surfaces them. A feature
must not have to import another feature to raise the right error, so the
vocabulary they share lives here instead.

"gated", "private" and "does not exist" call for three different actions from
whoever ran the ingest, and telling them only that it "failed" wastes their
time. ``NotFound`` and ``Unprocessable`` do the same job for the read/write
endpoints that are not about fetching: a router catches them and translates
to 404 or 422, and nothing else.
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


class AlreadyIngested(RuntimeError):
    """The model is already in the vault and the caller did not ask to refresh.

    Separate from NotFound and from IngestError because it is neither a missing
    thing nor a failure: the work was already done, and doing it again is a
    decision the caller has to make on purpose.
    """


class NotFound(RuntimeError):
    """A caller named a model, checkpoint, or benchmark the vault does not have.

    Not an :class:`IngestError`: fetching upstream never happened, or is not
    the point -- this is the store saying "nothing here", the same shape for
    every feature that reads by ID rather than fetches.
    """


class Unprocessable(RuntimeError):
    """The vault has the thing, but not what this operation needs from it.

    Distinct from :class:`NotFound` because the status code differs (422, not
    404): the model exists, but has no card to read, no checkpoint to extract
    for, or the request named something that is not there.
    """
