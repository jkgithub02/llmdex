import { Component, computed, inject, input, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { LlmdexService } from '../api/llmdex.service';
import type { Checkpoint } from '../api/model/checkpoint';
import type { ModelDoc } from '../api/model/modelDoc';
import type { Span } from '../api/model/span';
import { Field, formatBytes, formatCount } from '../field-state';
import { StateBadge } from '../state-badge';

/**
 * R6.3 / R6.4 - every field, including the null ones, each labelled with how we
 * came to know it.
 *
 * The four states are the product. A derived number came from `config.json`; an
 * extracted phrase was copied out of the card and names the section it came
 * from; "absent" means we looked and it was not stated; "unmeasured" means no
 * vendor publishes it and nobody has run it here yet. Flattening those into one
 * blank cell is the failure this view exists to prevent.
 */
@Component({
  selector: 'app-model-detail',
  imports: [RouterLink, StateBadge],
  template: `
    <a routerLink="/models" class="back">← models</a>

    @if (doc(); as model) {
      <header class="head">
        <span class="vendor">{{ vendor() }}</span>
        <h1 class="mono">{{ name() }}</h1>
      </header>

      @for (checkpoint of model.checkpoints ?? []; track checkpoint.repo + checkpoint.quantization) {
        <section class="checkpoint">
          <div class="ck-head">
            <span class="mono repo">{{ checkpoint.repo }}</span>
            @if (checkpoint.quantization) {
              <span class="tag mono">{{ checkpoint.quantization }}</span>
            }
            <span class="rev mono" title="the card revision this was built from (R1.3)">
              @{{ checkpoint.card_revision?.slice(0, 7) ?? 'unknown' }}
            </span>
          </div>

          <h2>Derived <span class="note">computed from config.json — never guessed</span></h2>
          <div class="grid">
            @for (field of derivedFields(checkpoint); track field.label) {
              <div class="field">
                <span class="key">{{ field.label }}</span>
                <span class="val mono" [class.null]="field.value === null">
                  {{ field.value ?? 'null' }}
                </span>
                <app-state-badge [state]="field.state" />
                @if (field.unreliable) {
                  <p class="why">{{ field.unreliable }}</p>
                }
              </div>
            }
          </div>

          <h2>Prose <span class="note">copied from the card — never generated</span></h2>
          @if (checkpoint.extracted; as extracted) {
            <p class="prov">
              extracted by <code>{{ extracted.model }}</code> on {{ extracted.extracted_on }}
              from card <code>{{ extracted.card_revision.slice(0, 7) }}</code>
            </p>
            <div class="grid">
              @for (field of extractedFields(checkpoint); track field.label) {
                <div class="field">
                  <span class="key">{{ field.label }}</span>
                  <span class="val" [class.null]="field.value === null">
                    {{ field.value ?? 'null' }}
                  </span>
                  <app-state-badge [state]="field.state" />
                  @if (field.source) {
                    <p class="why">§ {{ field.source }}</p>
                  }
                </div>
              }
            </div>
            @if (extracted.rejected?.length) {
              <div class="rejected">
                <h3>Rejected</h3>
                <p class="note">
                  The model proposed these. None of them appear in the card, so none were stored
                  (R3.2).
                </p>
                @for (item of extracted.rejected; track item.field + item.proposed) {
                  <p class="rej">
                    <code>{{ item.field }}</code>
                    <span class="proposed">{{ item.proposed }}</span>
                    <span class="reason mono">{{ item.reason }}</span>
                  </p>
                }
              </div>
            }
          } @else {
            <div class="cta">
              <p>Nobody has read this card yet.</p>
              <button (click)="extract(model.model_id)" [disabled]="extracting()">
                {{ extracting() ? 'Reading the card…' : 'Extract with the LLM' }}
              </button>
            </div>
            @if (extracting()) {
              <div class="bar"><span></span></div>
              <p class="note">
                The whole card goes to the model, which returns quotes. Only quotes we can find in
                the card are kept. About a minute.
              </p>
            }
          }

          <h2>Measured <span class="note">nothing here comes from a vendor</span></h2>
          @if (checkpoint.measured?.length) {
            @for (entry of checkpoint.measured; track $index) {
              <p class="mono">
                {{ entry.hardware }} · {{ entry.serving }} — TTFT {{ entry.ttft_ms ?? '—' }} ms
              </p>
            }
          } @else {
            <div class="field wide">
              <span class="key">latency, throughput, peak VRAM</span>
              <span class="val null">null</span>
              <app-state-badge state="unmeasured" />
              <p class="why">
                Properties of your deployment, not the model. No vendor publishes them.
              </p>
            </div>
          }
        </section>
      }
    } @else if (error(); as message) {
      <p class="error" role="alert">{{ message }}</p>
    } @else {
      <div class="bar"><span></span></div>
    }
  `,
  styles: `
    :host {
      display: block;
      padding: var(--space-8) var(--space-6);
      max-width: 64rem;
      margin: 0 auto;
    }
    .back {
      color: var(--fg-muted);
      font-size: 0.85rem;
    }
    .back:hover {
      color: var(--fg);
    }
    .head {
      margin: var(--space-4) 0 var(--space-6);
    }
    .vendor {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--fg-faint);
    }
    h1 {
      margin: 0.15rem 0 0;
      font-size: 1.45rem;
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    .checkpoint {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: var(--space-6);
      margin-bottom: var(--space-4);
    }
    .ck-head {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      flex-wrap: wrap;
      padding-bottom: var(--space-4);
      border-bottom: 1px solid var(--border);
    }
    .repo {
      font-size: 0.9rem;
      overflow-wrap: anywhere;
    }
    .tag {
      font-size: 0.7rem;
      padding: 0.1rem 0.4rem;
      border-radius: var(--radius-sm);
      background: var(--bg-sunken);
      border: 1px solid var(--border-strong);
    }
    .rev {
      margin-left: auto;
      font-size: 0.75rem;
      color: var(--fg-faint);
      cursor: help;
    }
    h2 {
      margin: var(--space-6) 0 var(--space-3);
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--fg-muted);
    }
    .note {
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0;
      color: var(--fg-faint);
      margin-left: var(--space-2);
      font-size: 0.78rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(19rem, 1fr));
      gap: var(--space-2);
    }
    .field {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.15rem var(--space-2);
      align-items: center;
      padding: var(--space-3);
      background: var(--bg-sunken);
      border: 1px solid var(--border);
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
      color: var(--fg-faint);
    }
    .val {
      font-size: 0.92rem;
      overflow-wrap: anywhere;
    }
    .val.null {
      color: var(--fg-faint);
      font-style: italic;
    }
    .why {
      grid-column: 1 / -1;
      margin: 0.2rem 0 0;
      font-size: 0.75rem;
      color: var(--fg-muted);
    }
    .prov {
      margin: 0 0 var(--space-3);
      font-size: 0.78rem;
      color: var(--fg-muted);
    }
    code {
      background: var(--bg-sunken);
      padding: 0.05rem 0.3rem;
      border-radius: 4px;
      border: 1px solid var(--border);
    }
    .rejected {
      margin-top: var(--space-4);
      padding: var(--space-3) var(--space-4);
      border-left: 2px solid var(--state-unmeasured);
      background: color-mix(in srgb, var(--state-unmeasured) 7%, transparent);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }
    .rejected h3 {
      margin: 0;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--state-unmeasured);
    }
    .rej {
      display: flex;
      gap: var(--space-2);
      align-items: baseline;
      flex-wrap: wrap;
      margin: var(--space-2) 0 0;
      font-size: 0.85rem;
    }
    .proposed {
      text-decoration: line-through;
      color: var(--fg-muted);
    }
    .reason {
      font-size: 0.7rem;
      color: var(--state-unmeasured);
    }
    .cta {
      display: flex;
      gap: var(--space-4);
      align-items: center;
      flex-wrap: wrap;
      padding: var(--space-4);
      border: 1px dashed var(--border-strong);
      border-radius: var(--radius-sm);
      color: var(--fg-muted);
    }
    button {
      background: var(--accent);
      color: #04250f;
      border: 0;
      border-radius: var(--radius);
      padding: 0 1.1rem;
      min-height: 44px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      transition:
        background 180ms ease,
        transform 120ms ease;
    }
    button:hover:not(:disabled) {
      background: #2ee06c;
    }
    button:active:not(:disabled) {
      transform: scale(0.98);
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .bar {
      height: 2px;
      background: var(--border);
      overflow: hidden;
      border-radius: 2px;
      margin: var(--space-3) 0;
    }
    .bar span {
      display: block;
      height: 100%;
      width: 35%;
      background: var(--accent);
      animation: slide 1.1s ease-in-out infinite;
    }
    @keyframes slide {
      0% {
        transform: translateX(-100%);
      }
      100% {
        transform: translateX(320%);
      }
    }
    .error {
      color: var(--danger);
    }
  `,
})
export class ModelDetail {
  private readonly api = inject(LlmdexService);

  /** A Hugging Face ID is `vendor/name`, so it arrives as two route segments. */
  readonly vendor = input.required<string>();
  readonly name = input.required<string>();

  protected readonly doc = signal<ModelDoc | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly extracting = signal(false);
  protected readonly modelId = computed(() => `${this.vendor()}/${this.name()}`);

  constructor() {
    queueMicrotask(() => this.load());
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
