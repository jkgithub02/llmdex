import { Component, inject, signal } from '@angular/core';

import { LlmdexService } from '../api/llmdex.service';
import type { Benchmark } from '../api/model/benchmark';

/**
 * R5.x - the benchmark documents in the vault.
 *
 * R5.5's caveat is shown unconditionally rather than tucked into a tooltip,
 * because two vendor-reported numbers side by side imply a comparability that
 * does not exist: they come from different harnesses, shot counts and agent
 * scaffolds, none of which the leaderboards require anyone to disclose.
 */
@Component({
  selector: 'app-benchmarks-page',
  host: { class: 'page' },
  template: `
    <header class="head">
      <h1>Benchmarks</h1>
      <p class="sub">{{ benchmarks()?.length ?? 0 }} documents</p>
    </header>

    <p class="caveat">
      Vendor-reported scores come from different harnesses and are not strictly comparable.
    </p>

    @if (benchmarks(); as rows) {
      @if (rows.length === 0) {
        <div class="empty">
          <p>No benchmark documents yet.</p>
          <p class="sub">
            Ingest creates a stub the first time a score refers to one. The explanation of what a
            benchmark measures is written by a person — the system will not generate it.
          </p>
        </div>
      } @else {
        <ul class="rows">
          @for (bench of rows; track bench.slug) {
            <li class="row">
              <div class="identity">
                <span class="name">{{ bench.name ?? bench.slug }}</span>
                <span class="slug mono">{{ bench.slug }}</span>
              </div>
              <span class="unit mono">{{ bench.unit ?? 'unit not stated' }}</span>
              @if (bench.unwritten) {
                <span class="stub">awaiting a human explanation</span>
              }
              @if (bench.explanation) {
                <p class="explanation">{{ bench.explanation }}</p>
              }
            </li>
          }
        </ul>
      }
    } @else {
      <div class="bar"><span></span></div>
    }
  `,
  styles: `
    .head {
      margin-bottom: var(--space-4);
    }
    h1 {
      margin: 0;
      font-size: 1.5rem;
      font-weight: 600;
      letter-spacing: -0.01em;
    }
    .caveat {
      border-left: 2px solid var(--state-absent);
      background: color-mix(in srgb, var(--state-absent) 8%, transparent);
      padding: var(--space-3) var(--space-4);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      color: var(--fg-muted);
      font-size: 0.85rem;
      margin: 0 0 var(--space-6);
    }
    .row {
      display: flex;
      align-items: baseline;
      gap: var(--space-4);
      flex-wrap: wrap;
      padding: var(--space-3) var(--space-4);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    .identity {
      display: flex;
      flex-direction: column;
    }
    .name {
      font-weight: 500;
    }
    .slug {
      font-size: 0.72rem;
      color: var(--fg-faint);
    }
    .unit {
      font-size: 0.8rem;
      color: var(--fg-muted);
    }
    .stub {
      margin-left: auto;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--state-absent);
      border: 1px solid currentColor;
      border-radius: 999px;
      padding: 0.1rem 0.45rem;
    }
    .explanation {
      flex-basis: 100%;
      margin: var(--space-2) 0 0;
      color: var(--fg-muted);
      font-size: 0.88rem;
    }
  `,
})
export class BenchmarksPage {
  private readonly api = inject(LlmdexService);
  protected readonly benchmarks = signal<Benchmark[] | null>(null);

  constructor() {
    this.api.listBenchmarksBenchmarksGet().subscribe({
      next: (rows) => this.benchmarks.set(rows),
      error: () => this.benchmarks.set([]),
    });
  }
}
