/**
 * A checkpoint, as rows a view can print.
 *
 * This is the only real logic in the frontend: everything else is markup or a
 * call to the generated client. It lives out here as plain functions rather
 * than methods on the detail component so it can be tested without rendering
 * anything or mocking HTTP -- the rules it encodes (a value we could not
 * compute is `absent`, not omitted; a quote names the section it came from) are
 * the ones worth a test.
 */

import type { Checkpoint } from '../api/model/checkpoint';
import type { HeadDim } from '../api/model/headDim';
import type { LayerComposition } from '../api/model/layerComposition';
import type { Span } from '../api/model/span';
import { formatBytes, formatCount } from '../format';
import type { Field } from '../provenance';

/**
 * R2.7 / R6.3 - every derived field, including the ones that came back null.
 *
 * All of them, not a chosen few: a reader who sees six rows cannot tell whether
 * the seventh was null or simply not shown, and that is exactly the ambiguity
 * the four states exist to remove.
 */
export function derivedFields(checkpoint: Checkpoint): Field[] {
  const derived = checkpoint.derived;
  const rows: Field[] = [
    // The class first: "MoE transformer" answers what someone actually asked,
    // where the raw model_type ("qwen3_moe") only names the config entry.
    { label: 'architecture', value: derived?.architecture_class ?? null, state: 'derived' },
    { label: 'model type', value: derived?.architecture ?? null, state: 'derived' },
    // Both carry the same reason: whatever made the total untrustworthy took
    // the active count with it, and "unavailable" alone would leave a reader
    // unable to tell a refused number from an unasked question (R2.7).
    {
      label: 'parameters (total)',
      value: formatCount(derived?.params?.total),
      state: 'derived',
      unreliable: derived?.params?.unreliable_reason ?? null,
    },
    {
      label: 'parameters (active)',
      value: formatCount(derived?.params?.active),
      state: 'derived',
      unreliable: derived?.params?.unreliable_reason ?? null,
    },
    { label: 'context length', value: formatCount(derived?.context_length), state: 'derived' },
    { label: 'hidden size', value: formatCount(derived?.hidden_size), state: 'derived' },
    { label: 'layers', value: formatCount(derived?.num_hidden_layers), state: 'derived' },
    layerRow(derived?.layers),
    {
      label: 'attention heads',
      value: formatCount(derived?.num_attention_heads),
      state: 'derived',
    },
    { label: 'KV heads', value: formatCount(derived?.num_key_value_heads), state: 'derived' },
    headDimRow(derived?.head_dim),
    { label: 'vocab size', value: formatCount(derived?.vocab_size), state: 'derived' },
    { label: 'dtype', value: derived?.torch_dtype ?? null, state: 'derived' },
    { label: 'weights on disk', value: formatBytes(derived?.weights?.bytes), state: 'derived' },
    {
      label: 'VRAM estimate',
      value: formatBytes(derived?.vram?.total_bytes),
      state: 'derived',
      unreliable: derived?.vram?.unreliable_reason ?? null,
    },
  ];
  return rows.map((row) => (row.value === null ? { ...row, state: 'absent' } : row));
}

/** R2.6a - the counts as published, never a ratio we assumed. */
function layerRow(layers: LayerComposition | undefined): Field {
  // The config often names the mechanism ("mamba", "linear attention"), and
  // "36 recurrent" throws that away. Fall back to the generic word only when
  // nothing said which it is.
  const parts = [
    [layers?.attention, 'attention'],
    [layers?.recurrent, layers?.recurrent_kind ?? 'recurrent'],
    [layers?.mlp_only, 'MLP-only'],
  ]
    .filter(([count]) => count)
    .map(([count, kind]) => `${count} ${kind}`);

  return {
    label: 'layer composition',
    value: parts.length ? parts.join(' · ') : null,
    state: 'derived',
    unreliable: layers?.unreliable_reason ?? null,
  };
}

/** R2.4a - when the config states a head_dim that its own numbers contradict, say so. */
function headDimRow(headDim: HeadDim | undefined): Field {
  const mismatch = headDim?.mismatch;
  return {
    label: 'head dim',
    value: formatCount(headDim?.value),
    state: 'derived',
    unreliable: mismatch
      ? `config states ${mismatch.explicit}; hidden size over heads gives ${mismatch.derived}`
      : null,
  };
}

export function extractedFields(checkpoint: Checkpoint): Field[] {
  const extracted = checkpoint.extracted;
  const quantization = extracted?.quantization;
  const rows: Field[] = [
    fromSpan('quantization format', quantization?.format),
    fromSpan('quantization method', quantization?.method),
    fromSpan('quantization scope', quantization?.scope),
    fromSpan('calibration', quantization?.calibration),
  ];
  for (const [engine, span] of Object.entries(extracted?.serving?.engines ?? {})) {
    rows.push(fromSpan(`serving · ${engine}`, span));
  }
  for (const row of extracted?.benchmarks ?? []) {
    rows.push(fromSpan(`benchmark · ${row.name.text}`, row.score));
  }
  return rows;
}

function fromSpan(label: string, span: Span | null | undefined): Field {
  if (!span) return { label, value: null, state: 'absent' };
  // R3.3 - a quote from before the first heading still has a place; say so
  // rather than leaving the reader to wonder where it came from.
  return { label, value: span.text, state: 'extracted', source: span.section || 'top of card' };
}
