/**
 * The reported benchmark scores of a checkpoint, as bars a view can print.
 *
 * Two sources feed this, and they are not the same kind of fact. A row copied
 * out of the card is a string — R3.1 forbids extraction from turning `"52.80"`
 * into `52.8`, so the number is parsed here, for the bar only, and `text` keeps
 * the characters the vendor actually wrote. A confirmed `BenchmarkScore` is
 * already a number and carries the URL it was reported at (R4.5a).
 *
 * Rows are grouped by task and one series is drawn per column header, because
 * a published table usually compares checkpoints — a BF16 sibling beside this
 * NVFP4 one. Nothing here decides which column is this repository: they are all
 * shown, labelled with the header the card used, which is the one presentation
 * that cannot attribute a number to the wrong model.
 *
 * Each task carries its own scale. Benchmarks are not comparable to each other
 * (R5.5) and they are not even on the same units: Nemotron's table puts GDPval
 * at 865 next to MMLU Pro at 81.94, and one shared axis rendered every
 * percentage benchmark on that card as a 1-to-10-percent sliver.
 *
 * A score that will not parse keeps its row and loses its bar. Dropping it
 * would let the chart imply the card reported fewer results than it did (R6.3).
 */

import type { Checkpoint } from '../../api/model/checkpoint';
import type { FieldState } from '../../shared/provenance';

export interface BenchmarkRow {
  /** The column header this score sat under, verbatim. Null for a single-column table. */
  variant: string | null;
  /** The score exactly as reported. */
  text: string;
  /** Bar length, 0-100, relative to the largest score shown. Null means unplottable. */
  percent: number | null;
  /** The card section it was copied from. */
  source: string | null;
  sourceUrl: string | null;
  state: FieldState;
}

export interface BenchmarkGroup {
  task: string;
  /** What this task's bars are drawn against, so a panel can print its scale. */
  domain: number;
  rows: BenchmarkRow[];
}

export function benchmarkGroups(checkpoint: Checkpoint): BenchmarkGroup[] {
  const flat: (BenchmarkRow & { task: string })[] = [
    // Confirmed first: a person had to pick the slug for one of these to exist
    // (R4.5b, R5.3), so it outranks the raw quote below it.
    ...(checkpoint.benchmarks ?? []).map((score) => ({
      task: score.slug,
      variant: null,
      text: `${score.score} ${score.unit}`.trim(),
      percent: score.score,
      source: null,
      sourceUrl: score.source_url ?? null,
      state: 'manual' as FieldState,
    })),
    ...(checkpoint.extracted_benchmarks?.rows ?? []).map((row) => ({
      task: plain(row.name.text),
      variant: row.variant ? plain(row.variant.text) : null,
      text: plain(row.score.text),
      percent: parseScore(plain(row.score.text)),
      source: row.score.section || 'top of card',
      sourceUrl: null,
      state: 'extracted' as FieldState,
    })),
  ];

  const groups = new Map<string, (BenchmarkRow & { value: number | null })[]>();
  for (const { task, ...row } of flat) {
    groups.set(task, [...(groups.get(task) ?? []), { ...row, value: row.percent }]);
  }

  return [...groups].map(([task, rows]) => {
    const domain = scaleFor(rows.map((row) => row.value));
    return {
      task,
      domain,
      rows: rows.map(({ value, ...row }) => ({
        ...row,
        percent: value === null ? null : (value / domain) * 100,
      })),
    };
  });
}

/**
 * The axis a task's bars are drawn against.
 *
 * Zero to a round bound above the largest score, never zero to the score
 * itself: scaling to the task's own maximum puts every top score at full width,
 * which would draw 9.28 out of 100 as a perfect result. Almost every benchmark
 * is scored out of 100, so that is the floor; anything above it — an Elo, a
 * token count — rounds up to the next 1, 2 or 5 times a power of ten, the way
 * an axis does.
 */
function scaleFor(values: (number | null)[]): number {
  const max = Math.max(0, ...values.filter((value): value is number => value !== null));
  if (max <= 100) return 100;
  const decade = 10 ** Math.floor(Math.log10(max));
  return [1, 2, 5, 10].map((step) => step * decade).find((bound) => bound >= max) ?? max;
}

/**
 * A cell without its markdown emphasis.
 *
 * Cards bold their table cells — DeepSeek-V2-Lite reports `**MMLU**` and
 * `**48.2**`. The asterisks are in the stored span because R3.1 keeps a span
 * verbatim against the card, and they are not part of what the vendor is
 * claiming, so they are neither printed nor parsed: `parseFloat('**48.2**')` is
 * NaN, which would quietly cost the row its bar. The stored span is untouched.
 */
function plain(text: string): string {
  return text.replace(/\*\*|__|`/g, '').trim();
}

/**
 * A reported score as one number, or null where it is not one number.
 *
 * Strict on purpose. `parseFloat` reads DeepSeek's `42.7 / 60.0` — HLE without
 * tools and with them, in one cell — as 42.7 and draws a bar the printed value
 * does not match. A bar that disagrees with the number beside it is worse than
 * no bar (R6.3), so the whole cell has to be a number, give or take the
 * thousands commas and a trailing unit sign.
 */
function parseScore(text: string): number | null {
  const cleaned = text.replace(/,/g, '').replace(/\s*%$/, '').trim();
  return /^-?\d+(\.\d+)?$/.test(cleaned) ? Number(cleaned) : null;
}
