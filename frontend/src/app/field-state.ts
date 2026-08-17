/**
 * R6.4 - every field is visibly one of four states, and R6.3 - null is
 * information and must not be hidden.
 *
 * The distinction the UI exists to preserve: a number computed from
 * `config.json` and a phrase copied out of a model card are different kinds of
 * fact, and "the card does not say" is different again from "nobody has
 * measured it yet". Most catalogues flatten all four into a blank cell.
 *
 * Each state carries a glyph as well as a colour, because colour alone must
 * never be the only thing distinguishing them.
 */
export type FieldState = 'derived' | 'extracted' | 'manual' | 'absent' | 'unmeasured';

export interface StateStyle {
  label: string;
  glyph: string;
  hint: string;
}

export const STATE: Record<FieldState, StateStyle> = {
  derived: {
    label: 'derived',
    glyph: '=',
    hint: 'Computed from config.json and the file listing. Never guessed.',
  },
  extracted: {
    label: 'extracted',
    glyph: '"',
    hint: 'Copied verbatim from the model card. Hover for the section it came from.',
  },
  manual: {
    label: 'by hand',
    glyph: '✎',
    hint: 'Entered by a person. Survives re-ingest.',
  },
  absent: {
    label: 'absent',
    glyph: '∅',
    hint: 'We looked. The card does not state it.',
  },
  unmeasured: {
    label: 'unmeasured',
    glyph: '~',
    hint: 'A property of your deployment. No vendor publishes it; nobody has run it here.',
  },
};

export interface Field {
  label: string;
  value: string | null;
  state: FieldState;
  /** R3.3 - the card section an extracted value was copied from. */
  source?: string | null;
  /** Why a derived value could not be computed (R2.6, R2.7). */
  unreliable?: string | null;
}

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
