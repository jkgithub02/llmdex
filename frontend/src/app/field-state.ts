/**
 * R6.4 - every field is visibly one of four states, and R6.3 - null is
 * information and must not be hidden.
 *
 * The distinction the UI exists to preserve: a number computed from
 * `config.json` and a phrase copied out of a model card are different kinds of
 * fact, and "the card does not say" is different again from "nobody has
 * measured it yet". Most catalogues flatten all four into a blank cell.
 */
export type FieldState = 'derived' | 'extracted' | 'manual' | 'absent' | 'unmeasured';

export const STATE_LABEL: Record<FieldState, string> = {
  derived: 'derived',
  extracted: 'extracted',
  manual: 'entered by hand',
  absent: 'absent from card',
  unmeasured: 'awaiting measurement',
};

export interface Field {
  label: string;
  value: string | null;
  state: FieldState;
  /** R3.3 - where an extracted value was copied from. */
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
  return value.toLocaleString('en-US');
}
