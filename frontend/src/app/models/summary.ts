/**
 * The generated summary, as sections a view can print.
 *
 * Kept out of the component for the same reason `fields.ts` is: the rules are
 * worth a test and rendering is not. The rule here is that an empty section
 * stays visible. The rest of this page shows a null because null is information
 * (R6.3); a summary with no limitations listed is the same kind of information,
 * and quietly dropping the heading would let a thin summary pass for a complete
 * one.
 */

import type { Summary } from '../api/model/summary';

export interface SummarySection {
  label: string;
  items: string[];
}

/**
 * The four lists, always all four, in the order a reader wants them: what is it,
 * then what is good, then what is bad, then what to do with it.
 */
export function summarySections(summary: Summary | null | undefined): SummarySection[] {
  if (!summary) return [];
  return [
    { label: 'what makes it different', items: summary.unique_points ?? [] },
    { label: 'strengths', items: summary.pros ?? [] },
    { label: 'limitations', items: summary.cons ?? [] },
    { label: 'good for', items: summary.use_cases ?? [] },
  ];
}

/** The site a source came from. The full URL is the link; this is its label. */
export function sourceHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    // Not a URL we can parse. Showing it raw beats dropping a source silently.
    return url;
  }
}
