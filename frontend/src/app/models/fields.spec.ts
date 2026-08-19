import type { Checkpoint } from '../api/model/checkpoint';
import type { Span } from '../api/model/span';
import { derivedFields, extractedFields } from './fields';

/**
 * The rules these cover are the ones the product rests on: a value we could not
 * compute is shown as `absent` rather than dropped (R2.7, R6.3), and a value we
 * did compute is never relabelled as a quote.
 */

function span(text: string, section = 'Quantization'): Span {
  return { text, start: 0, end: text.length, section };
}

function checkpoint(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { repo: 'Qwen/Qwen3-8B', ...overrides };
}

describe('derivedFields', () => {
  it('shows every field even when the checkpoint has nothing derived', () => {
    const fields = derivedFields(checkpoint());

    expect(fields.length).toBe(16);
    expect(fields.every((f) => f.value === null)).toBe(true);
    expect(fields.every((f) => f.state === 'absent')).toBe(true);
  });

  it('names a draft head shipped alongside the model, and never adds it in', () => {
    // R2.2 - NVFP4 Nemotron ships an MTP speculative-decoding head in the same
    // files. It is really on disk, so it is shown; it is not the model you
    // prompt, so it is not part of the total.
    const fields = derivedFields(
      checkpoint({
        derived: {
          params: { total: 31_577_940_288, auxiliary: 1_335_325_952, auxiliary_module: 'mtp' },
        },
      }),
    );
    const row = fields.find((f) => f.label.startsWith('parameters (draft head'))!;

    expect(row.label).toContain('mtp');
    expect(row.value).toBe('1.34B');
    expect(fields.find((f) => f.label === 'parameters (total)')!.value).toBe('31.58B');
  });

  it('says a model has no draft head rather than hiding the row', () => {
    const fields = derivedFields(checkpoint({ derived: { params: { total: 100 } } }));
    const row = fields.find((f) => f.label.startsWith('parameters (draft head'))!;

    expect(row.value).toBeNull();
    expect(row.state).toBe('absent');
  });

  it('leads with what kind of model it is, and keeps model_type as its own row', () => {
    const fields = derivedFields(
      checkpoint({ derived: { architecture: 'qwen3_moe', architecture_class: 'MoE transformer' } }),
    );

    expect(fields[0]).toEqual(
      jasmine.objectContaining({ label: 'architecture', value: 'MoE transformer' }),
    );
    expect(fields.find((f) => f.label === 'model type')!.value).toBe('qwen3_moe');
  });

  it('spells out a hybrid stack layer by layer', () => {
    const fields = derivedFields(
      checkpoint({
        derived: {
          num_hidden_layers: 52,
          layers: { family: 'nemotron_h', attention: 4, recurrent: 24, mlp_only: 24 },
        },
      }),
    );

    expect(fields.find((f) => f.label === 'layer composition')!.value).toBe(
      '4 attention · 24 recurrent · 24 MLP-only',
    );
  });

  it('reports an all-attention stack without inventing recurrent layers', () => {
    const fields = derivedFields(
      checkpoint({ derived: { layers: { family: 'transformer', attention: 80 } } }),
    );

    expect(fields.find((f) => f.label === 'layer composition')!.value).toBe('80 attention');
  });

  it('carries the reason a layer composition could not be read (R2.6c)', () => {
    const fields = derivedFields(
      checkpoint({
        derived: {
          layers: {
            family: 'unknown',
            unreliable_reason: 'config carries a state-space marker but no parseable composition',
          },
        },
      }),
    );
    const layers = fields.find((f) => f.label === 'layer composition')!;

    expect(layers.value).toBeNull();
    expect(layers.state).toBe('absent');
    expect(layers.unreliable).toContain('state-space marker');
  });

  it('surfaces a head_dim the config contradicts (R2.4a)', () => {
    const fields = derivedFields(
      checkpoint({
        derived: {
          head_dim: { value: 128, source: 'explicit', mismatch: { explicit: 128, derived: 125 } },
        },
      }),
    );
    const headDim = fields.find((f) => f.label === 'head dim')!;

    expect(headDim.value).toBe('128');
    expect(headDim.state).toBe('derived');
    expect(headDim.unreliable).toContain('125');
  });

  it('marks a computed value derived and an uncomputed one absent', () => {
    const fields = derivedFields(
      checkpoint({
        derived: {
          architecture_class: 'dense transformer',
          params: { total: 8_190_735_360, active: null },
          context_length: 32768,
        },
      }),
    );
    const by = (label: string) => fields.find((f) => f.label === label)!;

    expect(by('architecture')).toEqual(
      jasmine.objectContaining({ value: 'dense transformer', state: 'derived' }),
    );
    expect(by('parameters (total)')).toEqual(
      jasmine.objectContaining({ value: '8.19B', state: 'derived' }),
    );
    // Present in the payload, but null: the model is dense, so there is no
    // active count. That is absent, not a blank cell and not an omitted row.
    expect(by('parameters (active)')).toEqual(
      jasmine.objectContaining({ value: null, state: 'absent' }),
    );
  });

  it('carries the reason a VRAM estimate was withheld (R2.6)', () => {
    const fields = derivedFields(
      checkpoint({
        derived: {
          vram: {
            total_bytes: null,
            unreliable_reason: 'hybrid attention/SSM stack: KV cache is not comparable',
            assumptions: { context: 32768 },
          },
        },
      }),
    );
    const vram = fields.find((f) => f.label === 'VRAM estimate')!;

    expect(vram.value).toBeNull();
    expect(vram.state).toBe('absent');
    expect(vram.unreliable).toContain('hybrid');
  });
});

describe('extractedFields', () => {
  it('reports the four quantization fields as absent when nothing was extracted', () => {
    const fields = extractedFields(checkpoint());

    expect(fields.map((f) => f.label)).toEqual([
      'quantization format',
      'quantization method',
      'quantization scope',
      'calibration',
    ]);
    expect(fields.every((f) => f.state === 'absent')).toBe(true);
  });

  it('names the section a quote came from (R3.3)', () => {
    const fields = extractedFields(
      checkpoint({
        extracted: {
          card_revision: 'abc1234',
          extracted_on: '2026-08-17',
          model: 'vllm/Qwen3.5-122B',
          quantization: { format: span('GPTQ', 'Quantization details') },
        },
      }),
    );
    const format = fields.find((f) => f.label === 'quantization format')!;

    expect(format).toEqual(
      jasmine.objectContaining({
        value: 'GPTQ',
        state: 'extracted',
        source: 'Quantization details',
      }),
    );
  });

  it('says where a quote from before the first heading came from', () => {
    const fields = extractedFields(
      checkpoint({
        extracted: {
          card_revision: 'abc1234',
          extracted_on: '2026-08-17',
          model: 'vllm/Qwen3.5-122B',
          quantization: { method: span('AWQ', '') },
        },
      }),
    );

    expect(fields.find((f) => f.label === 'quantization method')!.source).toBe('top of card');
  });

  it('adds a row per serving engine', () => {
    const fields = extractedFields(
      checkpoint({
        extracted: {
          card_revision: 'abc1234',
          extracted_on: '2026-08-17',
          model: 'vllm/Qwen3.5-122B',
          serving: { engines: { vllm: span('vLLM >= 0.8', 'Deployment') } },
        },
      }),
    );

    // The results table is its own block now, read by its own agent.
    expect(fields.find((f) => f.label === 'serving · vllm')!.value).toBe('vLLM >= 0.8');
  });
});

describe('layer composition row', () => {
  it('names what the recurrent layers are when the config said', () => {
    // "24 recurrent" tells a reader less than the config actually stated.
    const fields = derivedFields(
      checkpoint({
        derived: {
          layers: {
            family: 'hybrid',
            attention: 12,
            recurrent: 36,
            recurrent_kind: 'linear attention',
          },
        },
      }),
    );

    expect(fields.find((f) => f.label === 'layer composition')!.value).toBe(
      '12 attention · 36 linear attention',
    );
  });

  it('falls back to "recurrent" when the kind was not stated', () => {
    const fields = derivedFields(
      checkpoint({ derived: { layers: { family: 'hybrid', attention: 4, recurrent: 28 } } }),
    );

    expect(fields.find((f) => f.label === 'layer composition')!.value).toBe(
      '4 attention · 28 recurrent',
    );
  });

  it('says so plainly when a model has no attention layers at all', () => {
    const fields = derivedFields(
      checkpoint({
        derived: {
          architecture_class: 'dense mamba',
          layers: { family: 'recurrent', attention: 0, recurrent: 24, recurrent_kind: 'mamba' },
        },
      }),
    );

    expect(fields.find((f) => f.label === 'layer composition')!.value).toBe('24 mamba');
  });
});

describe('parameter rows', () => {
  it('explains a count that could not be trusted rather than showing a bare gap', () => {
    // R2.7 - a quantized checkpoint's reported total counts packed bytes, and a
    // reader seeing only "unavailable" cannot tell that from "nobody looked".
    const fields = derivedFields(
      checkpoint({
        derived: {
          params: {
            total: null,
            active: null,
            is_moe: true,
            unreliable_reason: 'checkpoint is quantized to 4-bit weights',
          },
        },
      }),
    );

    const total = fields.find((f) => f.label === 'parameters (total)')!;
    const active = fields.find((f) => f.label === 'parameters (active)')!;
    expect(total.value).toBeNull();
    expect(total.unreliable).toBe('checkpoint is quantized to 4-bit weights');
    expect(active.unreliable).toBe('checkpoint is quantized to 4-bit weights');
  });

  it('says nothing extra when the counts are sound', () => {
    const fields = derivedFields(
      checkpoint({ derived: { params: { total: 30532122624, active: 3350000000, is_moe: true } } }),
    );

    expect(fields.find((f) => f.label === 'parameters (total)')!.unreliable).toBeNull();
  });
});
