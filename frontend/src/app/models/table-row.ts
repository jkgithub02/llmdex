/**
 * A model document flattened into one table row.
 *
 * The table shows the same facts as a card, and the point of pulling this out
 * as a function is that the flattening is where a dense grid quietly starts
 * lying: an empty cell reads as zero, and a withheld VRAM estimate reads as a
 * missing one. Both stay distinguishable here, and the view renders what it is
 * given (R6.3).
 */

import type { ModelDoc } from '../api/model/modelDoc';
import { formatBytes, formatCount } from '../format';
import { architectureTone, type Tone } from './architecture';

export interface TableRow {
  modelId: string;
  vendor: string;
  name: string;
  architecture: string;
  tone: Tone;
  params: string | null;
  context: string | null;
  vram: string | null;
  /** R2.6 - why the estimate was withheld, so the cell can say rather than gap. */
  vramReason: string | null;
  checkpoints: number;
}

export function tableRow(doc: ModelDoc): TableRow {
  const [vendor, ...rest] = doc.model_id.split('/');
  const derived = doc.checkpoints?.[0]?.derived;

  return {
    modelId: doc.model_id,
    vendor,
    name: rest.join('/'),
    architecture: derived?.architecture_class ?? 'unclassified',
    tone: architectureTone(derived?.architecture_class),
    params: formatCount(derived?.params?.total),
    context: formatCount(derived?.context_length),
    vram: formatBytes(derived?.vram?.total_bytes),
    vramReason: derived?.vram?.unreliable_reason ?? null,
    checkpoints: doc.checkpoints?.length ?? 0,
  };
}
