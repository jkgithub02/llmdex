import { Component, input, output } from '@angular/core';

import { AgentTrace } from '../../../shared/agent-trace';
import { StateBadge } from '../../../shared/state-badge';
import type { ModelDoc } from '../../../api/model/modelDoc';
import { benchmarkGroups } from '../benchmarks';
import { sourceHost } from '../summary';

/**
 * The Benchmarks tab: every reported score, grouped into small multiples (R5.5),
 * shown rather than tucked into a tooltip.
 */
@Component({
  selector: 'app-benchmarks-tab',
  imports: [AgentTrace, StateBadge],
  template: `
    <app-agent-trace [only]="modelId()" agent="benchmarks" />
    @for (
      checkpoint of model().checkpoints ?? [];
      track checkpoint.repo + checkpoint.quantization
    ) {
      <section class="pane">
        <p class="hint">copied from the card — different harnesses, not strictly comparable</p>
        <!-- Provenance once for the figure, not once per bar (same fact, repeated). -->
        @if (checkpoint.extracted_benchmarks; as read) {
          <p class="prov">
            <app-state-badge state="extracted" />
            every score copied verbatim by <code>{{ read.model }}</code> on
            {{ read.extracted_on }} from card
            <code>{{ read.card_revision.slice(0, 7) }}</code>
            @if (read.rejected?.length) {
              · {{ read.rejected!.length }} value(s) discarded, not found in the card
            }
          </p>
        }

        @if (benchmarkGroups(checkpoint); as groups) {
          @if (groups.length) {
            <!-- Small multiples: one panel per task, one bar per column the card
                 published. Identity is the label beside each bar, never a hue.
                 Which column is *this* repo is deliberately not decided here: a
                 wrong attribution is worse than none. -->
            <div class="figure">
              @for (group of groups; track $index) {
                <figure class="panel">
                  <!-- What the panel's bars are scored out of, so two panels out
                       of 100 and out of 1000 don't read as the same result. -->
                  <figcaption>
                    {{ group.task }}
                    <span class="scale mono">/{{ group.domain }}</span>
                  </figcaption>
                  @for (row of group.rows; track $index) {
                    <div
                      class="brow"
                      [title]="
                        (row.variant ?? 'as reported') +
                        ' — ' +
                        row.text +
                        (row.source ? ' · § ' + row.source : '')
                      "
                    >
                      <span class="bl" [class.unnamed]="!row.variant">{{
                        row.variant ?? 'as reported'
                      }}</span>
                      <span class="track">
                        <!-- An unreadable score keeps its place, loses its bar (R6.3). -->
                        @if (row.percent !== null) {
                          <span class="fill" [style.width.%]="row.percent"></span>
                        } @else {
                          <span class="noplot">not a number we can plot</span>
                        }
                      </span>
                      <span class="bv mono">{{ row.text }}</span>
                      <!-- Only where it differs from a copied row: R4.5a needs
                           the URL a confirmed score was reported at. -->
                      @if (row.state !== 'extracted') {
                        <span class="bsrc">
                          <app-state-badge [state]="row.state" />
                          @if (row.sourceUrl) {
                            <a [href]="row.sourceUrl" target="_blank" rel="noopener noreferrer">{{
                              sourceHost(row.sourceUrl)
                            }}</a>
                          }
                        </span>
                      }
                    </div>
                  }
                </figure>
              }
            </div>
          } @else {
            <div class="field wide">
              <span class="key">reported benchmark scores</span>
              <span class="val null">unavailable</span>
              <app-state-badge
                [state]="checkpoint.extracted_benchmarks ? 'absent' : 'unmeasured'"
              />
              <p class="why">
                @if (checkpoint.extracted_benchmarks) {
                  We read the card. It publishes no results table.
                } @else {
                  Nobody has read this card's results table yet.
                }
              </p>
            </div>
          }
        }

        <button class="ghost" (click)="rerun.emit('benchmarks')" [disabled]="busy()">Re-run</button>
      </section>
    }
  `,
  styles: `
    .pane {
      padding-top: var(--space-4);
    }
    .pane + .pane {
      border-top: 1px solid var(--sheet-border);
    }

    /* Small multiples: a panel per task, a bar per column the card compared.
       The bars are CSS -- one hue for all of them, since length already
       carries the magnitude. */
    .figure {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(19rem, 1fr));
      gap: var(--space-4) var(--space-6);
    }
    .panel {
      margin: 0;
      min-width: 0;
    }
    figcaption {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: var(--space-2);
      font-size: 0.78rem;
      font-weight: 600;
      padding-bottom: var(--space-1, 0.25rem);
      border-bottom: 1px solid var(--sheet-border);
      margin-bottom: var(--space-2);
      overflow-wrap: anywhere;
    }
    /* brow, not bar: ".page .bar" in styles.scss is the global loading indicator
       and reaches into a component regardless of its own scoping. */
    .scale {
      font-size: 0.68rem;
      font-weight: 400;
      color: var(--sheet-faint);
      white-space: nowrap;
    }
    .brow {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: baseline;
      /* 2px of surface between adjacent marks. */
      gap: 0 var(--space-2);
      padding-bottom: 2px;
    }
    .bl {
      font-size: 0.72rem;
      color: var(--sheet-muted);
      overflow-wrap: anywhere;
    }
    .bl.unnamed {
      font-style: italic;
      color: var(--sheet-faint);
    }
    .track {
      grid-column: 1;
      height: 0.4rem;
      border-radius: 999px;
      background: var(--sheet-sunken);
      overflow: hidden;
      align-self: center;
    }
    .fill {
      display: block;
      height: 100%;
      background: var(--tone);
      border-radius: 999px;
    }
    .noplot {
      display: block;
      font-size: 0.68rem;
      font-style: italic;
      line-height: 0.4rem;
      color: var(--sheet-faint);
      white-space: nowrap;
    }
    .bv {
      grid-column: 2;
      grid-row: 1 / span 2;
      align-self: center;
      font-size: 0.8rem;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .bsrc {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: var(--space-2);
      font-size: 0.7rem;
    }
    .bsrc a {
      color: var(--tone);
      text-decoration: underline;
    }

    .field {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.15rem var(--space-2);
      align-items: center;
      padding: var(--space-3);
      background: var(--sheet-sunken);
      border-radius: var(--radius-sm);
    }
    .field.wide {
      grid-column: 1 / -1;
    }
    .key {
      grid-column: 1 / -1;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--sheet-faint);
    }
    .val {
      font-size: 0.92rem;
      overflow-wrap: anywhere;
    }
    .val.null {
      color: var(--sheet-faint);
      font-style: italic;
    }
    .why {
      grid-column: 1 / -1;
      margin: 0.2rem 0 0;
      font-size: 0.75rem;
      color: var(--sheet-muted);
    }
    .hint,
    .prov {
      font-size: 0.78rem;
      color: var(--sheet-muted);
    }
    .hint {
      margin: 0 0 var(--space-3);
    }
    .prov {
      margin: var(--space-4) 0 var(--space-3);
    }
    code {
      background: var(--sheet-sunken);
      padding: 0.05rem 0.3rem;
      border-radius: 4px;
    }
    .pane button {
      background: var(--tone);
      color: #fff;
      border: none;
    }
    .pane button:hover:not(:disabled) {
      filter: brightness(1.08);
    }
    button.ghost {
      margin-top: var(--space-4);
      background: none;
      border: 1px solid var(--sheet-border);
      color: var(--sheet-muted);
      font-size: 0.78rem;
      padding: 0.3rem 0.7rem;
    }
    button.ghost:hover:not(:disabled) {
      filter: none;
      color: var(--sheet-fg);
      border-color: var(--sheet-faint);
    }
  `,
})
export class BenchmarksTab {
  readonly model = input.required<ModelDoc>();
  readonly modelId = input.required<string>();
  readonly busy = input.required<boolean>();
  readonly rerun = output<string>();

  protected readonly benchmarkGroups = benchmarkGroups;
  protected readonly sourceHost = sourceHost;
}
