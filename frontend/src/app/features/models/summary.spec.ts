import type { Summary } from '../../api/model/summary';
import { sourceHost, summarySections } from './summary';

/**
 * The summary is generated rather than copied, so what the view must never do is
 * let it read like the rest of the sheet: an empty list is the generator finding
 * nothing to say, not a fact about the model, and it has to look different from
 * a derived null.
 */

function summary(overrides: Partial<Summary> = {}): Summary {
  return {
    overview: 'Qwen3-8B is a dense 8B model from Alibaba.',
    generated_by: 'vllm/some-model',
    generated_on: '2026-08-17',
    ...overrides,
  };
}

describe('summarySections', () => {
  it('always returns the four sections in a fixed order', () => {
    const sections = summarySections(summary());

    expect(sections.map((s) => s.label)).toEqual([
      'what makes it different',
      'strengths',
      'limitations',
      'good for',
    ]);
  });

  it('carries the items it was given', () => {
    const sections = summarySections(
      summary({
        unique_points: ['Thinking and non-thinking modes'],
        pros: ['Fits a single 24GB card'],
        cons: ['Short context'],
        use_cases: ['Local assistants', 'Agentic tool use'],
      }),
    );

    expect(sections[0].items).toEqual(['Thinking and non-thinking modes']);
    expect(sections[3].items).toEqual(['Local assistants', 'Agentic tool use']);
  });

  it('keeps an empty section rather than hiding it', () => {
    // Hiding it would leave a reader unable to tell "no limitations were found"
    // from "limitations were never asked for".
    const sections = summarySections(summary({ pros: ['Fast'] }));

    expect(sections.find((s) => s.label === 'limitations')).toEqual({
      label: 'limitations',
      items: [],
    });
  });

  it('has nothing to show for a model that has never been summarised', () => {
    expect(summarySections(null)).toEqual([]);
    expect(summarySections(undefined)).toEqual([]);
  });
});

describe('sourceHost', () => {
  it('names the site rather than printing the whole URL', () => {
    expect(sourceHost('https://openrouter.ai/qwen/qwen3-8b')).toBe('openrouter.ai');
  });

  it('drops a leading www so the list reads evenly', () => {
    expect(sourceHost('https://www.together.ai/models/qwen3-8b')).toBe('together.ai');
  });

  it('falls back to the raw string rather than throwing on something unparseable', () => {
    expect(sourceHost('not a url')).toBe('not a url');
  });
});
