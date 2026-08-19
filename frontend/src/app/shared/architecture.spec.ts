import { architecturePills, architectureTone, layerSegments } from './architecture';

/**
 * The two axes `architecture_class` already encodes, pulled apart for display.
 * The rule these cover: nothing here may invent a classification the backend
 * did not make. An unclassified model reads as unclassified, not as a default.
 */

describe('architecturePills', () => {
  it('splits the two axes the class already encodes', () => {
    expect(architecturePills('MoE hybrid (mamba)')).toEqual(['MoE', 'hybrid (mamba)']);
    expect(architecturePills('dense transformer')).toEqual(['dense', 'transformer']);
  });

  it('keeps a multi-word mixture together', () => {
    expect(architecturePills('MoE hybrid (lightning attention)')).toEqual([
      'MoE',
      'hybrid (lightning attention)',
    ]);
  });

  it('handles a model with no attention layers, whose class names the mechanism', () => {
    expect(architecturePills('dense mamba')).toEqual(['dense', 'mamba']);
    expect(architecturePills('dense RWKV')).toEqual(['dense', 'RWKV']);
  });

  it('says unclassified rather than guessing when the backend could not classify', () => {
    expect(architecturePills(null)).toEqual(['unclassified']);
    expect(architecturePills(undefined)).toEqual(['unclassified']);
  });
});

describe('architectureTone', () => {
  it('colours by what the layers are made of, not by how the FFN is routed', () => {
    // A dense and an MoE transformer are the same shape of model to look at.
    expect(architectureTone('dense transformer')).toBe('transformer');
    expect(architectureTone('MoE transformer')).toBe('transformer');
  });

  it('separates hybrids and attention-free models from transformers', () => {
    expect(architectureTone('dense hybrid (mamba)')).toBe('hybrid');
    expect(architectureTone('MoE hybrid (linear attention)')).toBe('hybrid');
    expect(architectureTone('dense mamba')).toBe('recurrent');
    expect(architectureTone('dense RWKV')).toBe('recurrent');
  });

  it('has its own tone for a model nothing could be read from', () => {
    expect(architectureTone(null)).toBe('unknown');
  });
});

describe('layerSegments', () => {
  it('returns one segment per kind, in stack order, with counts', () => {
    const segments = layerSegments({
      family: 'nemotron_h',
      attention: 4,
      recurrent: 24,
      mlp_only: 24,
      recurrent_kind: 'mamba',
    });

    expect(segments).toEqual([
      { kind: 'attention', count: 4, label: '4 attention' },
      { kind: 'recurrent', count: 24, label: '24 mamba' },
      { kind: 'mlp', count: 24, label: '24 MLP-only' },
    ]);
  });

  it('names the recurrent layers generically when the config did not say', () => {
    const segments = layerSegments({ family: 'jamba', attention: 4, recurrent: 28 });

    expect(segments[1].label).toBe('28 recurrent');
  });

  it('drops kinds with no layers rather than drawing an empty band', () => {
    const segments = layerSegments({ family: 'transformer', attention: 36 });

    expect(segments).toEqual([{ kind: 'attention', count: 36, label: '36 attention' }]);
  });

  it('has nothing to draw when the composition could not be read', () => {
    expect(layerSegments(undefined)).toEqual([]);
    expect(layerSegments({ family: 'unknown', unreliable_reason: 'no config' })).toEqual([]);
  });
});
