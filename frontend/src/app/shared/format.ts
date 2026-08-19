/**
 * Turning values into the strings the views print.
 *
 * Every one of these returns null rather than an empty string or a dash when it
 * has nothing: what a null means is the caller's decision to render (R6.3), not
 * something a formatter should quietly settle by printing a blank.
 */

export function formatBytes(bytes: number | null | undefined): string | null {
  if (bytes === null || bytes === undefined) return null;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

export function formatCount(value: number | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return value.toLocaleString('en-US');
}

/** A Hugging Face ID is `vendor/name`, and the router holds it as two segments. */
export function routeFor(modelId: string): string[] {
  const [vendor, ...rest] = modelId.split('/');
  return ['/models', vendor, rest.join('/')];
}

/**
 * What went wrong, in the words the backend used.
 *
 * FastAPI puts the sentence a person should read in `detail`, and those
 * sentences are the point: "gated repository", "no model card to read", "set
 * LLMDEX_LLM_BASE_URL". Falling back to the HttpErrorResponse's own message
 * would replace all of them with "Http failure response for /api/...".
 */
export function errorMessage(err: unknown): string {
  const detail = (err as { error?: { detail?: string } })?.error?.detail;
  return detail ?? (err as { message?: string })?.message ?? 'request failed';
}
