import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';

import { LlmdexService } from '../api/llmdex.service';
import type { ModelDoc } from '../api/model/modelDoc';
import { errorMessage, formatBytes, formatCount, routeFor } from '../format';
import { StateBadge } from '../state-badge';

/**
 * R6.2 - everything in the vault, plus the box that fills it.
 *
 * The VRAM column shows an explicit "withheld" where the estimate is null, not a
 * blank. For a hybrid architecture we cannot cost correctly, refusing to print a
 * number is the product working (R2.6) — so it says so, and says why on hover.
 */
@Component({
  selector: 'app-models-page',
  imports: [FormsModule, RouterLink, StateBadge],
  host: { class: 'page' },
  template: `
    <header class="head">
      <div>
        <h1>Models</h1>
        <p class="sub">
          {{ count() }} in the vault · every field carries how it came to be known
        </p>
      </div>
    </header>

    <form class="ingest" (submit)="ingest($event)">
      <label class="sr-only" for="model-id">Hugging Face model ID or URL</label>
      <input
        id="model-id"
        class="mono"
        [(ngModel)]="modelId"
        name="modelId"
        [disabled]="busy()"
        placeholder="Qwen/Qwen3-8B  or  https://huggingface.co/…"
        autocomplete="off"
        spellcheck="false"
      />
      <button type="submit" [disabled]="busy() || !modelId.trim()">
        {{ busy() ? 'Fetching…' : 'Ingest' }}
      </button>
    </form>
    @if (busy()) {
      <div class="bar"><span></span></div>
    }
    @if (error(); as message) {
      <p class="error" role="alert">{{ message }}</p>
    }

    @if (models(); as rows) {
      @if (rows.length === 0) {
        <div class="empty">
          <p>Nothing here yet.</p>
          <p class="sub">Paste a model ID above. Ingest reads config.json and the file listing.</p>
        </div>
      } @else {
        <ul class="rows">
          @for (row of rows; track row.model_id) {
            <li>
              <a [routerLink]="route(row.model_id)" class="row">
                <div class="identity">
                  <span class="vendor">{{ vendor(row.model_id) }}</span>
                  <span class="name mono">{{ name(row.model_id) }}</span>
                </div>

                <dl class="facts">
                  <div>
                    <dt>params</dt>
                    <dd class="mono">{{ params(row) ?? '—' }}</dd>
                  </div>
                  <div>
                    <dt>context</dt>
                    <dd class="mono">{{ context(row) ?? '—' }}</dd>
                  </div>
                  <div>
                    <dt>vram</dt>
                    <dd class="mono" [class.withheld]="!vram(row)" [title]="vramReason(row) ?? ''">
                      {{ vram(row) ?? 'withheld' }}
                    </dd>
                  </div>
                  <div>
                    <dt>checkpoints</dt>
                    <dd class="mono">{{ row.checkpoints?.length ?? 0 }}</dd>
                  </div>
                </dl>

                <app-state-badge [state]="proseState(row)" />
              </a>
            </li>
          }
        </ul>
      }
    } @else {
      <div class="bar"><span></span></div>
    }
  `,
  styles: `
    /* Wider than the shared default: four fact columns and a badge per row. */
    :host(.page) {
      max-width: 68rem;
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: var(--space-6);
    }
    h1 {
      margin: 0;
      font-size: 1.5rem;
      font-weight: 600;
      letter-spacing: -0.01em;
    }
    .ingest {
      display: flex;
      gap: var(--space-2);
      margin-bottom: var(--space-6);
    }
    input {
      flex: 1;
      background: var(--bg-sunken);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--fg);
      padding: 0.65rem 0.85rem;
      font-size: 0.9rem;
      transition: border-color 180ms ease;
    }
    input::placeholder {
      color: var(--fg-faint);
    }
    input:hover:not(:disabled) {
      border-color: var(--border-strong);
    }
    .row {
      display: grid;
      grid-template-columns: minmax(12rem, 1fr) auto auto;
      gap: var(--space-4);
      align-items: center;
      padding: var(--space-3) var(--space-4);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      transition:
        border-color 180ms ease,
        background 180ms ease,
        transform 120ms ease;
    }
    .row:hover {
      border-color: var(--border-strong);
      background: var(--bg-raised);
    }
    .row:active {
      transform: scale(0.995);
    }
    .identity {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .name {
      font-size: 0.95rem;
      font-weight: 500;
      overflow-wrap: anywhere;
    }
    .facts {
      display: flex;
      gap: var(--space-6);
      margin: 0;
    }
    .facts div {
      display: flex;
      flex-direction: column;
    }
    dt {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--fg-faint);
    }
    dd {
      margin: 0;
      font-size: 0.9rem;
    }
    .withheld {
      color: var(--state-absent);
      cursor: help;
    }
    @media (max-width: 760px) {
      .row {
        grid-template-columns: 1fr;
        gap: var(--space-3);
      }
      .facts {
        flex-wrap: wrap;
        gap: var(--space-4);
      }
    }
  `,
})
export class ModelsPage {
  private readonly api = inject(LlmdexService);

  protected modelId = '';
  protected readonly models = signal<ModelDoc[] | null>(null);
  protected readonly busy = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly count = computed(() => this.models()?.length ?? 0);

  constructor() {
    this.refresh();
  }

  protected route = routeFor;

  protected vendor(modelId: string): string {
    return modelId.split('/')[0];
  }

  protected name(modelId: string): string {
    return modelId.split('/').slice(1).join('/');
  }

  protected refresh(): void {
    this.api.listModelsModelsGet().subscribe({
      next: (rows) => this.models.set(rows),
      error: (err) => this.error.set(errorMessage(err)),
    });
  }

  protected ingest(event: Event): void {
    event.preventDefault();
    const id = this.modelId.trim();
    if (!id) return;
    this.busy.set(true);
    this.error.set(null);
    this.api.ingestModelIngestPost({ model_id: id }).subscribe({
      next: () => {
        this.busy.set(false);
        this.modelId = '';
        this.refresh();
      },
      error: (err) => {
        this.busy.set(false);
        this.error.set(errorMessage(err));
      },
    });
  }

  protected params(doc: ModelDoc): string | null {
    return formatCount(doc.checkpoints?.[0]?.derived?.params?.total ?? null);
  }

  protected context(doc: ModelDoc): string | null {
    return formatCount(doc.checkpoints?.[0]?.derived?.context_length ?? null);
  }

  protected vram(doc: ModelDoc): string | null {
    return formatBytes(doc.checkpoints?.[0]?.derived?.vram?.total_bytes ?? null);
  }

  /** R2.6 - when the estimate is withheld, say what made it unreliable. */
  protected vramReason(doc: ModelDoc): string | null {
    return doc.checkpoints?.[0]?.derived?.vram?.unreliable_reason ?? null;
  }

  protected proseState(doc: ModelDoc): 'extracted' | 'manual' | 'absent' {
    const checkpoint = doc.checkpoints?.[0];
    if (checkpoint?.extracted) return 'extracted';
    if (checkpoint?.manual?.quantization || checkpoint?.manual?.serving) return 'manual';
    return 'absent';
  }
}
