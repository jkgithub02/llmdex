/**
 * R6.4 - every field is visibly one of four states, and R6.3 - null is
 * information and must not be hidden.
 *
 * The distinction the UI exists to preserve: a number computed from
 * `config.json` and a phrase copied out of a model card are different kinds of
 * fact, and "the card does not say" is different again from "nobody has
 * measured it yet". Most catalogues flatten all four into a blank cell.
 *
 * Each state is named in words, because colour alone must never be the only
 * thing distinguishing them.
 */
export type FieldState = 'derived' | 'extracted' | 'manual' | 'absent' | 'unmeasured';

export interface StateStyle {
  label: string;
  hint: string;
}

export const STATE: Record<FieldState, StateStyle> = {
  derived: {
    label: 'computed',
    hint: 'Computed from config.json and the file listing. Never guessed.',
  },
  extracted: {
    label: 'quoted',
    hint: 'Copied verbatim from the model card. Hover for the section it came from.',
  },
  manual: {
    label: 'reviewed',
    hint: 'Entered by a person. Survives re-ingest.',
  },
  absent: {
    label: 'not stated',
    hint: 'We looked. The card does not state it.',
  },
  unmeasured: {
    label: 'not run yet',
    // Was worded for deployment latency, which was the only thing wearing this
    // state while the Measured tab existed. It now marks anything nobody has
    // produced yet, which is a different fact from the card not stating it.
    hint: 'Nobody has produced this yet. Different from the card not stating it.',
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
