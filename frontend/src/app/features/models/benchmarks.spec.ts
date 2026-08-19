import type { Checkpoint } from '../../api/model/checkpoint';
import type { Span } from '../../api/model/span';
import { benchmarkGroups } from './benchmarks';

/**
 * The rules worth pinning: a reported score keeps the characters the card used,
 * a score we cannot turn into a number is still listed (R6.3) — it just gets no
 * bar — and a table comparing two checkpoints stays two labelled series rather
 * than being silently merged into one.
 */

function span(text: string, section = 'Benchmarks'): Span {
  return { text, start: 0, end: text.length, section };
}

function checkpoint(rows: unknown[]): Checkpoint {
  return {
    repo: 'nvidia/X-NVFP4',
    extracted_benchmarks: {
      card_revision: 'abc',
      extracted_on: '2026-08-18',
      model: 'test',
      rows,
    },
  } as Checkpoint;
}

describe('benchmarkGroups', () => {
  it('is empty when nobody has read the table', () => {
    expect(benchmarkGroups({ repo: 'a/b' })).toEqual([]);
  });

  it('keeps one series per column, labelled with the column header', () => {
    const groups = benchmarkGroups(
      checkpoint([
        { name: span('MMLU Pro'), score: span('81.94'), variant: span('...-BF16') },
        { name: span('MMLU Pro'), score: span('81.62'), variant: span('...-NVFP4') },
        { name: span('GPQA Diamond'), score: span('75.44'), variant: span('...-BF16') },
      ]),
    );

    expect(groups.map((g) => g.task)).toEqual(['MMLU Pro', 'GPQA Diamond']);
    expect(groups[0].rows.map((r) => r.variant)).toEqual(['...-BF16', '...-NVFP4']);
    expect(groups[0].rows.map((r) => r.text)).toEqual(['81.94', '81.62']);
  });

  it('gives each benchmark its own scale, not one shared with the card', () => {
    // The bug this replaces: Nemotron's table has GDPval at 865 beside MMLU Pro
    // at 81.94. Scaled against the card's largest score, every percentage
    // benchmark rendered between 1% and 10% wide — a column of identical
    // slivers. Benchmarks are not comparable to each other (R5.5), so they do
    // not share an axis.
    const groups = benchmarkGroups(
      checkpoint([
        { name: span('MMLU Pro'), score: span('81.94') },
        { name: span('GDPval-AA-V2'), score: span('865') },
      ]),
    );

    expect(groups[0].domain).toBe(100);
    expect(groups[0].rows[0].percent).toBeCloseTo(81.94, 2);
    expect(groups[1].domain).toBe(1000);
    expect(groups[1].rows[0].percent).toBeCloseTo(86.5, 2);
  });

  it('keeps a low-scoring benchmark low rather than filling its panel', () => {
    // Scaling to the task's own maximum would put every top score at full
    // width, so 9.28 out of 100 would look like a perfect result.
    const groups = benchmarkGroups(checkpoint([{ name: span('t3-bench'), score: span('9.28') }]));

    expect(groups[0].domain).toBe(100);
    expect(groups[0].rows[0].percent).toBeCloseTo(9.28, 2);
  });

  it('rounds an above-100 scale up to a readable bound', () => {
    const groups = benchmarkGroups(
      checkpoint([
        { name: span('Elo'), score: span('1450') },
        { name: span('Elo'), score: span('1200') },
      ]),
    );

    expect(groups[0].domain).toBe(2000);
    expect(groups[0].rows[0].percent).toBeCloseTo(72.5, 2);
  });

  it('will not plot a cell holding two numbers', () => {
    // DeepSeek reports `42.7 / 60.0` for HLE, one figure without tools and one
    // with. parseFloat takes 42.7 and draws a bar the printed value does not
    // match, which is worse than no bar at all (R6.3).
    const groups = benchmarkGroups(
      checkpoint([{ name: span('HLE (wo / w tools)'), score: span('42.7 / 60.0') }]),
    );

    expect(groups[0].rows[0].text).toBe('42.7 / 60.0');
    expect(groups[0].rows[0].percent).toBeNull();
  });

  it('still plots a score written with a percent sign or thousands comma', () => {
    const groups = benchmarkGroups(
      checkpoint([
        { name: span('MMLU'), score: span('52.80%') },
        { name: span('Tokens'), score: span('1,450') },
      ]),
    );

    expect(groups[0].rows[0].percent).toBeCloseTo(52.8, 2);
    expect(groups[1].rows[0].percent).toBeCloseTo(72.5, 2);
  });

  it('lists a score it cannot plot rather than dropping it', () => {
    const groups = benchmarkGroups(
      checkpoint([
        { name: span('MMLU Pro'), score: span('81.94') },
        { name: span('SWE-bench'), score: span('not reported') },
      ]),
    );

    expect(groups.length).toBe(2);
    expect(groups[1].rows[0].text).toBe('not reported');
    expect(groups[1].rows[0].percent).toBeNull();
  });

  it('prints a bolded cell as the name and number the card meant', () => {
    // Cards bold their cells: DeepSeek-V2-Lite reports `**MMLU**` and
    // `**DeepSeek 7B (Dense)**`. The asterisks are markdown syntax, part of the
    // stored span because R3.1 keeps the span verbatim, and not part of what
    // the vendor is claiming — so they are not printed, and not parsed either:
    // parseFloat('**48.2**') is NaN, which would silently drop the bar.
    const groups = benchmarkGroups(
      checkpoint([
        { name: span('**MMLU**'), score: span('**48.2**'), variant: span('**DeepSeek 7B**') },
      ]),
    );

    expect(groups[0].task).toBe('MMLU');
    expect(groups[0].rows[0].variant).toBe('DeepSeek 7B');
    expect(groups[0].rows[0].text).toBe('48.2');
    expect(groups[0].rows[0].percent).toBeCloseTo(48.2, 2);
  });

  it('records the section a row was copied from', () => {
    const groups = benchmarkGroups(
      checkpoint([{ name: span('MMLU Pro'), score: span('81.94', 'Reasoning') }]),
    );

    expect(groups[0].rows[0].source).toBe('Reasoning');
    expect(groups[0].rows[0].state).toBe('extracted');
  });

  it('carries a confirmed score with the URL it was reported at (R4.5a)', () => {
    const groups = benchmarkGroups({
      repo: 'a/b',
      benchmarks: [
        {
          slug: 'mmlu',
          score: 85.2,
          unit: '%',
          provenance: 'vendor',
          source_url: 'https://huggingface.co/a/b',
        },
      ],
    } as Checkpoint);

    expect(groups[0].task).toBe('mmlu');
    expect(groups[0].rows[0]).toEqual(
      jasmine.objectContaining({
        text: '85.2 %',
        percent: 85.2,
        sourceUrl: 'https://huggingface.co/a/b',
        state: 'manual',
      }),
    );
  });
});
