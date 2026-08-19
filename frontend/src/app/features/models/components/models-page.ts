import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AgentStream } from '../../../shared/agent-stream';
import { LlmdexService } from '../../../api/llmdex.service';
import type { ModelDoc } from '../../../api/model/modelDoc';
import { errorMessage } from '../../../shared/format';
import { ModelsGrid } from './models-grid';
import { ModelsTable } from './models-table';

/**
 * R6.2 - everything in the vault, plus the box that fills it.
 *
 * The VRAM column shows an explicit "withheld" where the estimate is null, not a
 * blank. For a hybrid architecture we cannot cost correctly, refusing to print a
 * number is the product working (R2.6) — so it says so, and says why on hover.
 */
const VIEW_KEY = 'llmdex.models.view';

@Component({
  selector: 'app-models-page',
  imports: [FormsModule, ModelsGrid, ModelsTable],
  host: { class: 'page' },
  template: `
    <header class="head">
      <div>
        <h1>Models</h1>
        <p class="sub">{{ count() }} in the vault · every field carries how it came to be known</p>
      </div>

      <div class="views" role="group" aria-label="layout">
        <button type="button" [class.on]="view() === 'grid'" (click)="setView('grid')">
          Cards
        </button>
        <button type="button" [class.on]="view() === 'table'" (click)="setView('table')">
          Table
        </button>
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
        @if (view() === 'table') {
          <app-models-table
            [rows]="rows"
            [confirming]="confirming()"
            [deleting]="deleting()"
            (remove)="remove($event)"
            (confirmChange)="confirming.set($event)"
          />
        } @else {
          <app-models-grid
            [rows]="rows"
            [confirming]="confirming()"
            [deleting]="deleting()"
            (remove)="remove($event)"
            (confirmChange)="confirming.set($event)"
          />
        }
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
    .views {
      display: flex;
      gap: 0;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .views button {
      background: none;
      border: none;
      border-radius: 0;
      color: var(--fg-muted);
      font-size: 0.78rem;
      padding: 0.3rem 0.7rem;
    }
    .views button.on {
      background: var(--surface);
      color: var(--fg);
    }
  `,
})
export class ModelsPage {
  private readonly api = inject(LlmdexService);
  private readonly stream = inject(AgentStream);

  protected modelId = '';
  protected readonly models = signal<ModelDoc[] | null>(null);
  protected readonly busy = signal(false);
  protected readonly error = signal<string | null>(null);
  /** Which card is asking to be confirmed, and which is mid-delete. Shared by
   * the grid and table views, so it lives here rather than in either. */
  protected readonly confirming = signal<string | null>(null);
  protected readonly deleting = signal<string | null>(null);
  protected readonly count = computed(() => this.models()?.length ?? 0);

  constructor() {
    this.refresh();
  }

  /**
   * Is this model already in the vault? Compared case-insensitively and against
   * the normalised ID, because the box accepts a full Hub URL and the backend
   * stores the bare `vendor/name`.
   */
  private knows(raw: string): boolean {
    const wanted = raw
      .trim()
      .replace(/^https?:\/\/huggingface\.co\//i, '')
      .replace(/\/+$/, '')
      .toLowerCase();
    return (this.models() ?? []).some((doc) => doc.model_id.toLowerCase() === wanted);
  }

  /**
   * Cards or table. Remembered because it is a working preference, not a
   * navigation state: a reader who prefers the table wants it on every visit,
   * not just until the next reload.
   */
  protected readonly view = signal<'grid' | 'table'>(
    localStorage.getItem(VIEW_KEY) === 'table' ? 'table' : 'grid',
  );

  protected setView(next: 'grid' | 'table'): void {
    this.view.set(next);
    localStorage.setItem(VIEW_KEY, next);
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
    // Decided before the POST: refresh() below is async, so by the time the
    // handler runs the list already contains the model either way.
    const alreadyKnown = this.knows(id);
    this.api.ingestModelIngestPost({ model_id: id }).subscribe({
      next: () => {
        this.busy.set(false);
        const ingested = id;
        this.modelId = '';
        this.refresh();
        // Only on a model's first ingest. A re-ingest refreshes the derived
        // facts from config.json, and must not spend LLM calls doing it or
        // overwrite a summary somebody regenerated on purpose (R6.5) -- the
        // same rule summarise_after_first_ingest keeps on the backend.
        if (!alreadyKnown) {
          this.stream.start(ingested, ['about', 'prose', 'benchmarks']);
        }
      },
      error: (err) => {
        this.busy.set(false);
        this.error.set(errorMessage(err));
      },
    });
  }

  /**
   * Remove a model from the vault.
   *
   * Destructive, so it asks first. It is not irreversible though: the vault is a
   * git repository and the removal lands as a commit, so a mistake is recovered
   * with `git revert` rather than by re-ingesting and losing the summary.
   */
  protected remove(modelId: string): void {
    this.deleting.set(modelId);
    this.error.set(null);
    this.api.deleteModelModelsModelIdDelete(modelId).subscribe({
      next: () => {
        this.deleting.set(null);
        this.confirming.set(null);
        this.refresh();
      },
      error: (err) => {
        this.deleting.set(null);
        this.confirming.set(null);
        this.error.set(errorMessage(err));
      },
    });
  }
}
