import { Component, computed, input } from '@angular/core';
import MarkdownIt from 'markdown-it';

/**
 * An answer, rendered.
 *
 * A model comparing two checkpoints reaches for a table, and a table read as
 * raw pipes and dashes is the answer arriving unreadable.
 *
 * No sanitizer here on purpose: Angular sanitizes `[innerHTML]` itself,
 * stripping scripts, event handlers and javascript: URLs. Reaching for
 * `bypassSecurityTrustHtml` would be turning that off, and this renders text a
 * language model wrote from vendor prose -- exactly the input not to trust.
 *
 * `html: false` for the same reason: a card that embeds raw HTML should not be
 * able to smuggle it through the model into our DOM.
 */
const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
});

@Component({
  selector: 'app-markdown',
  template: `<div class="md" [innerHTML]="rendered()"></div>`,
  styles: `
    .md {
      font-size: 0.88rem;
      line-height: 1.6;
      overflow-wrap: anywhere;
    }
    /* No margin above the first block or below the last, so a turn's
       spacing comes from the turn, not from whatever markdown starts it. */
    .md ::ng-deep p:first-child,
    .md ::ng-deep h1:first-child,
    .md ::ng-deep h2:first-child,
    .md ::ng-deep h3:first-child {
      margin-top: 0;
    }
    .md ::ng-deep p:last-child,
    .md ::ng-deep ul:last-child,
    .md ::ng-deep ol:last-child,
    .md ::ng-deep pre:last-child,
    .md ::ng-deep .table-wrap:last-child {
      margin-bottom: 0;
    }
    .md ::ng-deep p,
    .md ::ng-deep ul,
    .md ::ng-deep ol {
      margin: 0 0 var(--space-3);
    }
    .md ::ng-deep ul,
    .md ::ng-deep ol {
      padding-left: var(--space-4);
    }
    .md ::ng-deep li {
      margin-bottom: var(--space-1);
    }
    .md ::ng-deep h1,
    .md ::ng-deep h2,
    .md ::ng-deep h3,
    .md ::ng-deep h4 {
      margin: var(--space-4) 0 var(--space-2);
      font-size: 0.92rem;
      font-weight: 600;
      line-height: 1.3;
    }
    .md ::ng-deep code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em;
      background: var(--sheet-sunken);
      border-radius: var(--radius-sm);
      padding: 0.1em 0.35em;
    }
    .md ::ng-deep pre {
      background: var(--sheet-sunken);
      border-radius: var(--radius-sm);
      padding: var(--space-3);
      overflow-x: auto;
      margin: 0 0 var(--space-3);
    }
    .md ::ng-deep pre code {
      background: none;
      padding: 0;
    }
    .md ::ng-deep blockquote {
      margin: 0 0 var(--space-3);
      padding-left: var(--space-3);
      border-left: 2px solid var(--sheet-border);
      color: var(--sheet-muted);
    }

    /* Tables are why this component exists: a model comparing two checkpoints
       reaches for one, and the panel is narrow, so it scrolls on its own
       rather than widening the page. */
    .md ::ng-deep .table-wrap {
      overflow-x: auto;
      margin: 0 0 var(--space-3);
    }
    .md ::ng-deep table {
      border-collapse: collapse;
      font-size: 0.8rem;
      width: 100%;
    }
    .md ::ng-deep th,
    .md ::ng-deep td {
      border: 1px solid var(--sheet-border);
      padding: var(--space-1) var(--space-2);
      text-align: left;
      white-space: nowrap;
    }
    .md ::ng-deep th {
      background: var(--sheet-sunken);
      font-weight: 600;
    }
    .md ::ng-deep a {
      color: var(--type-transformer);
      text-decoration: underline;
    }
  `,
})
export class Markdown {
  readonly text = input.required<string>();

  protected readonly rendered = computed(() =>
    // The wrapper is what lets a wide table scroll inside the panel instead of
    // pushing the layout sideways.
    md
      .render(this.text())
      .replace(/<table>/g, '<div class="table-wrap"><table>')
      .replace(/<\/table>/g, '</table></div>'),
  );
}
