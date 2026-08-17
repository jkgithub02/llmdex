import { Component, inject, signal } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatProgressBarModule } from '@angular/material/progress-bar';

import { LlmdexService } from '../api/llmdex.service';
import type { Benchmark } from '../api/model/benchmark';

/**
 * R5.x - the benchmark documents in the vault.
 *
 * R5.5's caveat is shown unconditionally, because two vendor-reported numbers
 * side by side imply a comparability that does not exist: they come from
 * different harnesses, shot counts and scaffolds, none of which the leaderboards
 * require anyone to disclose.
 */
@Component({
  selector: 'app-benchmarks-page',
  imports: [MatCardModule, MatProgressBarModule],
  template: `
    <p class="caveat">
      Vendor-reported scores come from different harnesses and are not strictly comparable (R5.5).
    </p>

    @if (benchmarks(); as rows) {
      @if (rows.length === 0) {
        <p class="empty">
          No benchmark documents yet. Ingest creates a stub the first time a score refers to one
          (R5.3); the explanation is written by a person, never generated (R5.2).
        </p>
      } @else {
        @for (bench of rows; track bench.slug) {
          <mat-card>
            <mat-card-header>
              <mat-card-title>{{ bench.name ?? bench.slug }}</mat-card-title>
              <mat-card-subtitle>
                {{ bench.unit ?? 'unit not stated' }}
                @if (bench.unwritten) {
                  · stub, awaiting a human explanation
                }
              </mat-card-subtitle>
            </mat-card-header>
            @if (bench.explanation) {
              <mat-card-content>{{ bench.explanation }}</mat-card-content>
            }
          </mat-card>
        }
      }
    } @else {
      <mat-progress-bar mode="indeterminate" />
    }
  `,
  styles: `
    :host {
      display: block;
      padding: 1.5rem;
      max-width: 60rem;
      margin: 0 auto;
    }
    .caveat {
      border-left: 3px solid #8a6d1f;
      padding-left: 0.75rem;
      opacity: 0.85;
    }
    .empty {
      opacity: 0.7;
    }
    mat-card {
      margin-bottom: 1rem;
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
