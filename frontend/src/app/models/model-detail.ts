import { Component, computed, inject, input, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { RouterLink } from '@angular/router';

import { LlmdexService } from '../api/llmdex.service';
import type { Checkpoint } from '../api/model/checkpoint';
import type { ModelDoc } from '../api/model/modelDoc';
import type { Span } from '../api/model/span';
import { Field, STATE_LABEL, formatBytes, formatCount } from '../field-state';

/**
 * R6.3 / R6.4 - every field, including the null ones, each labelled with how we
 * came to know it.
 *
 * The four states are the product. A derived number came from `config.json`; an
 * extracted phrase was copied out of the card and names the section it came
 * from; "absent from card" means we looked and it was not stated; "awaiting
 * measurement" means no vendor publishes it and nobody has run it here yet.
 */
@Component({
  selector: 'app-model-detail',
  imports: [
    RouterLink,
    MatButtonModule,
    MatCardModule,
    MatChipsModule,
    MatIconModule,
    MatProgressBarModule,
    MatTooltipModule,
  ],
  template: `
    <a routerLink="/models" class="back"><mat-icon>arrow_back</mat-icon> All models</a>

    @if (doc(); as model) {
      <h1>{{ model.model_id }}</h1>
      @for (checkpoint of model.checkpoints ?? []; track checkpoint.repo + checkpoint.quantization) {
        <mat-card>
          <mat-card-header>
            <mat-card-title>
              {{ checkpoint.repo }}
              @if (checkpoint.quantization) {
                <span class="quant">{{ checkpoint.quantization }}</span>
              }
            </mat-card-title>
            <mat-card-subtitle>
              card {{ checkpoint.card_revision?.slice(0, 7) ?? 'unknown' }} · ingested
              {{ checkpoint.ingested ?? '—' }}
            </mat-card-subtitle>
          </mat-card-header>

          <mat-card-content>
            <h3>Derived <small>computed from config.json — never guessed</small></h3>
            <dl>
              @for (field of derivedFields(checkpoint); track field.label) {
                <dt>{{ field.label }}</dt>
                <dd>
                  <span class="value" [class.null]="field.value === null">
                    {{ field.value ?? 'null' }}
                  </span>
                  <span class="state {{ field.state }}">{{ label(field.state) }}</span>
                  @if (field.unreliable) {
                    <span class="reason">{{ field.unreliable }}</span>
                  }
                </dd>
              }
            </dl>

            <h3>
              Prose
              <small>copied from the card, or entered by hand — never generated</small>
            </h3>
            @if (checkpoint.extracted; as extracted) {
              <p class="provenance">
                extracted by <code>{{ extracted.model }}</code> on {{ extracted.extracted_on }} from
                card {{ extracted.card_revision.slice(0, 7) }}
              </p>
              <dl>
                @for (field of extractedFields(checkpoint); track field.label) {
                  <dt>{{ field.label }}</dt>
                  <dd>
                    <span class="value" [class.null]="field.value === null">
                      {{ field.value ?? 'null' }}
                    </span>
                    <span class="state {{ field.state }}">{{ label(field.state) }}</span>
                    @if (field.source) {
                      <span class="source" matTooltip="R3.3 — the card section this was copied from">
                        § {{ field.source }}
                      </span>
                    }
                  </dd>
                }
              </dl>
              @if (extracted.rejected?.length) {
                <div class="rejected">
                  <h4>
                    Rejected <small>the model proposed these; none appear in the card (R3.2)</small>
                  </h4>
                  @for (item of extracted.rejected; track item.field + item.proposed) {
                    <p>
                      <code>{{ item.field }}</code> — {{ item.proposed }}
                      <span class="reason">{{ item.reason }}</span>
                    </p>
                  }
                </div>
              }
            } @else {
              <p class="empty">
                Nobody has read this card yet.
                <button
                  mat-flat-button
                  (click)="extract(model.model_id)"
                  [disabled]="extracting()"
                >
                  Extract with the LLM
                </button>
              </p>
              @if (extracting()) {
                <mat-progress-bar mode="indeterminate" />
                <p class="hint">Reading the whole card. This takes about a minute.</p>
              }
            }

            <h3>Measured <small>nothing here comes from a vendor (R4.2)</small></h3>
            @if (checkpoint.measured?.length) {
              @for (entry of checkpoint.measured; track $index) {
                <p>{{ entry.hardware }} · {{ entry.serving }} — TTFT {{ entry.ttft_ms ?? '—' }} ms</p>
              }
            } @else {
              <p class="empty">
                <span class="state unmeasured">{{ label('unmeasured') }}</span>
                Latency, throughput and real peak VRAM are properties of your deployment. They stay
                null until someone runs a benchmark.
              </p>
            }
          </mat-card-content>
        </mat-card>
      }
    } @else if (error(); as message) {
      <p class="error"><mat-icon>error_outline</mat-icon> {{ message }}</p>
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
    .back {
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      margin-bottom: 1rem;
    }
    h1 {
      font-size: 1.4rem;
      word-break: break-all;
    }
    h3 {
      margin: 1.5rem 0 0.5rem;
      font-size: 1rem;
    }
    h3 small,
    h4 small {
      font-weight: 400;
      opacity: 0.6;
      margin-left: 0.5rem;
    }
    dl {
      display: grid;
      grid-template-columns: 14rem 1fr;
      gap: 0.35rem 1rem;
      margin: 0;
    }
    dt {
      opacity: 0.75;
    }
    dd {
      margin: 0;
      display: flex;
      gap: 0.5rem;
      align-items: baseline;
      flex-wrap: wrap;
    }
    .value.null {
      opacity: 0.45;
      font-style: italic;
    }
    .state {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0.1rem 0.4rem;
      border-radius: 0.6rem;
      border: 1px solid currentColor;
      opacity: 0.8;
    }
    .state.derived {
      color: #2e6f4e;
    }
    .state.extracted {
      color: #1e5aa8;
    }
    .state.manual {
      color: #7a4ba0;
    }
    .state.absent {
      color: #8a6d1f;
    }
    .state.unmeasured {
      color: #9a3b3b;
    }
    .source,
    .reason,
    .provenance,
    .hint {
      font-size: 0.8rem;
      opacity: 0.7;
    }
    .quant {
      margin-left: 0.5rem;
      font-size: 0.8rem;
      opacity: 0.7;
    }
    .rejected {
      margin-top: 1rem;
      padding: 0.5rem 1rem;
      border-left: 3px solid #9a3b3b;
    }
    .empty {
      opacity: 0.8;
      display: flex;
      gap: 0.5rem;
      align-items: center;
      flex-wrap: wrap;
    }
    .error {
      color: var(--mat-sys-error);
    }
  `,
})
export class ModelDetail {
  private readonly api = inject(LlmdexService);

  /** Route params. A Hugging Face ID is `vendor/name`, so it arrives in two parts. */
  readonly vendor = input.required<string>();
  readonly name = input.required<string>();

  protected readonly doc = signal<ModelDoc | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly extracting = signal(false);
  protected readonly modelId = computed(() => `${this.vendor()}/${this.name()}`);

  constructor() {
    queueMicrotask(() => this.load());
  }

  protected label(state: keyof typeof STATE_LABEL): string {
    return STATE_LABEL[state];
  }

  protected load(): void {
    this.api.getModelModelsModelIdGet(this.modelId()).subscribe({
      next: (doc) => this.doc.set(doc),
      error: (err) => this.error.set(this.message(err)),
    });
  }

  protected extract(modelId: string): void {
    this.extracting.set(true);
    this.error.set(null);
    this.api.extractModelModelsModelIdExtractPost(modelId).subscribe({
      next: (doc) => {
        this.extracting.set(false);
        this.doc.set(doc);
      },
      error: (err) => {
        this.extracting.set(false);
        this.error.set(this.message(err));
      },
    });
  }

  /** R2.7 - a field that could not be computed is shown as absent, not omitted. */
  protected derivedFields(checkpoint: Checkpoint): Field[] {
    const derived = checkpoint.derived;
    const rows: Field[] = [
      { label: 'architecture', value: derived?.architecture ?? null, state: 'derived' },
      { label: 'parameters (total)', value: formatCount(derived?.params?.total), state: 'derived' },
      {
        label: 'parameters (active)',
        value: formatCount(derived?.params?.active),
        state: 'derived',
      },
      { label: 'context length', value: formatCount(derived?.context_length), state: 'derived' },
      { label: 'KV heads', value: formatCount(derived?.num_key_value_heads), state: 'derived' },
      { label: 'weights on disk', value: formatBytes(derived?.weights?.bytes), state: 'derived' },
      {
        label: 'VRAM estimate',
        value: formatBytes(derived?.vram?.total_bytes),
        state: 'derived',
        unreliable: derived?.vram?.unreliable_reason ?? null,
      },
    ];
    return rows.map((row) => (row.value === null ? { ...row, state: 'absent' } : row));
  }

  protected extractedFields(checkpoint: Checkpoint): Field[] {
    const extracted = checkpoint.extracted;
    const quantization = extracted?.quantization;
    const rows: Field[] = [
      this.fromSpan('quantization format', quantization?.format),
      this.fromSpan('quantization method', quantization?.method),
      this.fromSpan('quantization scope', quantization?.scope),
      this.fromSpan('calibration', quantization?.calibration),
    ];
    for (const [engine, span] of Object.entries(extracted?.serving?.engines ?? {})) {
      rows.push(this.fromSpan(`serving · ${engine}`, span));
    }
    for (const row of extracted?.benchmarks ?? []) {
      rows.push(this.fromSpan(`benchmark · ${row.name.text}`, row.score));
    }
    return rows;
  }

  private fromSpan(label: string, span: Span | null | undefined): Field {
    if (!span) return { label, value: null, state: 'absent' };
    return { label, value: span.text, state: 'extracted', source: span.section || 'top of card' };
  }

  private message(err: unknown): string {
    const detail = (err as { error?: { detail?: string }; message?: string })?.error?.detail;
    return detail ?? (err as { message?: string })?.message ?? 'request failed';
  }
}
