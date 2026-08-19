/**
 * `architecture_class` as a view can wear it: two pills and a colour.
 *
 * The backend already answers "what kind of model is this" as a single string
 * across two axes -- how the FFN is routed, and what the layers are made of.
 * Nothing here re-derives that; it only pulls the two halves apart so they can
 * be shown as separate pills, the way a Pokedex shows Fire and Flying.
 *
 * The one rule: a model the backend could not classify reads as `unclassified`.
 * Picking a likely default here would put back exactly the guess `derive.py`
 * was fixed to stop making.
 */

import type { LayerComposition } from '../api/model/layerComposition';

export type Tone = 'transformer' | 'hybrid' | 'recurrent' | 'unknown';

/** `MoE hybrid (mamba)` -> `['MoE', 'hybrid (mamba)']`. */
export function architecturePills(architectureClass: string | null | undefined): string[] {
  if (!architectureClass) return ['unclassified'];
  // Only the first space splits: the mixture itself can be several words.
  const at = architectureClass.indexOf(' ');
  if (at === -1) return [architectureClass];
  return [architectureClass.slice(0, at), architectureClass.slice(at + 1)];
}

/**
 * Colour follows the layout axis, not the routing one: a dense and an MoE
 * transformer are the same shape of thing to look at, while a hybrid and a
 * transformer are not.
 */
export function architectureTone(architectureClass: string | null | undefined): Tone {
  if (!architectureClass) return 'unknown';
  const layout = architecturePills(architectureClass)[1] ?? '';
  if (layout.startsWith('hybrid')) return 'hybrid';
  if (layout === 'transformer') return 'transformer';
  return 'recurrent';
}

export interface LayerSegment {
  kind: 'attention' | 'recurrent' | 'mlp';
  count: number;
  label: string;
}

/**
 * The layer stack as bands to draw, in stack order.
 *
 * This is the detail page's headline graphic, and it is the composition as
 * published rather than a ratio anyone assumed (R2.6a). A model whose
 * composition could not be read draws nothing at all -- an empty strip is the
 * honest picture of an unknown stack, and a full one would be a claim.
 */
export function layerSegments(layers: LayerComposition | null | undefined): LayerSegment[] {
  if (!layers || layers.family === 'unknown' || layers.unreliable_reason) return [];

  const recurrentName = layers.recurrent_kind ?? 'recurrent';
  return (
    [
      { kind: 'attention', count: layers.attention ?? 0, name: 'attention' },
      { kind: 'recurrent', count: layers.recurrent ?? 0, name: recurrentName },
      { kind: 'mlp', count: layers.mlp_only ?? 0, name: 'MLP-only' },
    ] as const
  )
    .filter((band) => band.count > 0)
    .map((band) => ({
      kind: band.kind,
      count: band.count,
      label: `${band.count} ${band.name}`,
    }));
}
