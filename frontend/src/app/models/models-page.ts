import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';

import { AgentStream } from '../agents/agent-stream';
import { AgentTrace } from '../agents/agent-trace';
import { LlmdexService } from '../api/llmdex.service';
import type { ModelDoc } from '../api/model/modelDoc';
import { errorMessage, formatBytes, formatCount, routeFor } from '../format';
import { StateBadge } from '../state-badge';
import { architecturePills, architectureTone } from './architecture';
import { LayerStrip } from './layer-strip';
import { tableRow } from './table-row';

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
  imports: [FormsModule, RouterLink, StateBadge, LayerStrip, AgentTrace],
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
    <app-agent-trace />
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
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">model</th>
                  <th scope="col">architecture</th>
                  <th scope="col" class="num">params</th>
                  <th scope="col" class="num">context</th>
                  <th scope="col" class="num">vram</th>
                  <th scope="col" class="num">ckpts</th>
                  <th scope="col"><span class="sr-only">actions</span></th>
                </tr>
              </thead>
              <tbody>
                @for (row of rows; track row.model_id) {
                  @let cells = tableRow(row);
                  <tr>
                    <th scope="row">
                      <a [routerLink]="route(cells.modelId)">
                        <span class="t-vendor">{{ cells.vendor }}/</span
                        ><span class="mono">{{ cells.name }}</span>
                      </a>
                    </th>
                    <td>
                      <span class="chip" [attr.data-tone]="cells.tone">{{
                        cells.architecture
                      }}</span>
                    </td>
                    <td class="num mono">{{ cells.params ?? '—' }}</td>
                    <td class="num mono">{{ cells.context ?? '—' }}</td>
                    <!-- Withheld and unknown are different facts (R2.6): the
                         reason is the title, so the cell is never a bare gap. -->
                    <td
                      class="num mono"
                      [class.withheld-cell]="cells.vramReason"
                      [title]="cells.vramReason ?? ''"
                    >
                      {{ cells.vram ?? (cells.vramReason ? 'withheld' : '—') }}
                    </td>
                    <td class="num mono">{{ cells.checkpoints }}</td>
                    <td class="row-actions">
                      @if (confirming() === cells.modelId) {
                        <button type="button" class="yes" (click)="remove(cells.modelId)">
                          {{ deleting() === cells.modelId ? '…' : 'Delete' }}
                        </button>
                        <button type="button" class="no" (click)="confirming.set(null)">
                          Keep
                        </button>
                      } @else {
                        <button
                          type="button"
                          class="del-inline"
                          [attr.aria-label]="'delete ' + cells.modelId"
                          (click)="confirming.set(cells.modelId)"
                        >
                          ✕
                        </button>
                      }
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        } @else {
          <ul class="dex-grid">
            @for (row of rows; track row.model_id) {
              <li>
                <!-- Outside the anchor on purpose: a button inside a link is
                   invalid HTML, and the click would navigate before deleting. -->
                @if (confirming() === row.model_id) {
                  <div class="confirm">
                    <span>Delete?</span>
                    <button type="button" class="yes" (click)="remove(row.model_id)">
                      {{ deleting() === row.model_id ? '…' : 'Delete' }}
                    </button>
                    <button type="button" class="no" (click)="confirming.set(null)">Keep</button>
                  </div>
                } @else {
                  <button
                    type="button"
                    class="del"
                    [attr.aria-label]="'delete ' + row.model_id"
                    (click)="confirming.set(row.model_id)"
                  >
                    ✕
                  </button>
                }
                <a [routerLink]="route(row.model_id)" class="card" [attr.data-tone]="tone(row)">
                  <div class="card-top">
                    <span class="vendor">{{ vendor(row.model_id) }}</span>
                    <app-state-badge [state]="proseState(row)" />
                  </div>

                  <h2 class="mono">{{ name(row.model_id) }}</h2>

                  <div class="pills">
                    @for (pill of pills(row); track pill) {
                      <span class="pill">{{ pill }}</span>
                    }
                  </div>

                  <app-layer-strip [layers]="row.checkpoints?.[0]?.derived?.layers" />

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
                      <dd
                        class="mono"
                        [class.withheld]="!vram(row)"
                        [title]="vramReason(row) ?? ''"
                      >
                        {{ vram(row) ?? 'withheld' }}
                      </dd>
                    </div>
                  </dl>
                </a>
              </li>
            }
          </ul>
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

    .table-wrap {
      /* Wide content scrolls inside its own box rather than the page. */
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th,
    td {
      text-align: left;
      padding: var(--space-3) var(--space-4);
      white-space: nowrap;
    }
    thead th {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--fg-faint);
      font-weight: 600;
      background: var(--bg-sunken);
      border-bottom: 1px solid var(--border);
    }
    tbody tr {
      border-bottom: 1px solid var(--border);
    }
    tbody tr:last-child {
      border-bottom: none;
    }
    tbody tr:hover {
      background: var(--surface);
    }
    tbody th {
      font-weight: 500;
    }
    .t-vendor {
      color: var(--fg-faint);
    }
    .num {
      text-align: right;
    }
    /* The tone travels with the row as a chip, so the architecture is as
       glanceable in the table as the card colour makes it in the grid. */
    .chip {
      display: inline-block;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      font-size: 0.72rem;
      color: #fff;
    }
    .chip[data-tone='transformer'] {
      background: var(--type-transformer);
    }
    .chip[data-tone='hybrid'] {
      background: var(--type-hybrid);
    }
    .chip[data-tone='recurrent'] {
      background: var(--type-recurrent);
    }
    .chip[data-tone='unknown'] {
      background: var(--type-unknown);
    }
    .withheld-cell {
      color: var(--state-absent);
      cursor: help;
    }
    .row-actions {
      text-align: right;
    }
    .row-actions button {
      font-size: 0.7rem;
      padding: 0.15rem 0.5rem;
    }
    .del-inline {
      background: none;
      border: 1px solid var(--border-strong);
      color: var(--fg-faint);
    }
    .del-inline:hover {
      background: var(--danger);
      border-color: var(--danger);
      color: #fff;
    }

    .dex-grid li {
      position: relative;
    }
    .del,
    .confirm {
      position: absolute;
      top: var(--space-3);
      right: var(--space-3);
      z-index: 1;
    }
    .del {
      background: rgb(0 0 0 / 0.18);
      border: none;
      color: rgb(255 255 255 / 0.75);
      border-radius: 999px;
      width: 1.5rem;
      height: 1.5rem;
      padding: 0;
      font-size: 0.8rem;
      line-height: 1;
      opacity: 0.45;
      transition: opacity 140ms ease;
    }
    /* Dimmed rather than hidden. Revealing it on hover would keep the
       destructive control out of the way, but there is no hover on a touch
       screen, and a button you cannot reach without a mouse fails R6.7. */
    li:hover .del,
    .del:focus-visible {
      opacity: 1;
    }
    .del:hover {
      background: var(--danger);
      color: #fff;
    }
    .confirm {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      background: var(--bg-sunken);
      border: 1px solid var(--danger);
      border-radius: var(--radius-sm);
      padding: 0.2rem 0.4rem;
      font-size: 0.72rem;
      color: var(--fg);
    }
    .confirm button {
      font-size: 0.7rem;
      padding: 0.1rem 0.45rem;
    }
    .confirm .yes {
      background: var(--danger);
      color: #fff;
    }
    .confirm .no {
      background: none;
      border: 1px solid var(--border-strong);
      color: var(--fg-muted);
    }
    .dex-grid {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
      gap: var(--space-4);
    }
    /* One tone per architecture layout, same mapping the detail hero uses, so a
       model is the same colour wherever you meet it. */
    .card[data-tone='transformer'] {
      --tone: var(--type-transformer);
    }
    .card[data-tone='hybrid'] {
      --tone: var(--type-hybrid);
    }
    .card[data-tone='recurrent'] {
      --tone: var(--type-recurrent);
    }
    .card[data-tone='unknown'] {
      --tone: var(--type-unknown);
    }
    .card {
      display: block;
      background: var(--tone);
      background-image: radial-gradient(
        120% 80% at 85% 0%,
        rgb(255 255 255 / 0.18),
        transparent 60%
      );
      border-radius: var(--radius);
      padding: var(--space-4);
      color: #fff;
      transition:
        transform 140ms ease,
        box-shadow 140ms ease;
    }
    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 24px rgb(0 0 0 / 0.35);
    }
    .card:active {
      transform: translateY(0);
    }
    .card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-2);
      /* Room for the delete control, which floats over this corner from
         outside the anchor. */
      padding-right: 1.9rem;
    }
    .card .vendor {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgb(255 255 255 / 0.85);
    }
    .card h2 {
      margin: var(--space-1) 0 var(--space-3);
      font-size: 1.05rem;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .pills {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
      margin-bottom: var(--space-4);
    }
    .pill {
      background: rgb(255 255 255 / 0.22);
      border: 1px solid rgb(255 255 255 / 0.25);
      border-radius: 999px;
      padding: 0.1rem 0.6rem;
      font-size: 0.7rem;
      font-weight: 500;
    }
    .facts {
      display: flex;
      gap: var(--space-4);
      margin: var(--space-4) 0 0;
      flex-wrap: wrap;
    }
    .facts div {
      display: flex;
      flex-direction: column;
    }
    dt {
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: rgb(255 255 255 / 0.7);
    }
    dd {
      margin: 0;
      font-size: 0.85rem;
      font-weight: 500;
    }
    .withheld {
      color: #ffe9a8;
      cursor: help;
    }
    @media (max-width: 760px) {
      .dex-grid {
        grid-template-columns: 1fr;
      }
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
  /** Which card is asking to be confirmed, and which is mid-delete. */
  protected readonly confirming = signal<string | null>(null);
  protected readonly deleting = signal<string | null>(null);
  protected readonly count = computed(() => this.models()?.length ?? 0);

  constructor() {
    this.refresh();
  }

  protected route = routeFor;
  protected tableRow = tableRow;

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

  protected pills(doc: ModelDoc): string[] {
    return architecturePills(doc.checkpoints?.[0]?.derived?.architecture_class);
  }

  protected tone(doc: ModelDoc): string {
    return architectureTone(doc.checkpoints?.[0]?.derived?.architecture_class);
  }

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
        const ingested = id;
        this.modelId = '';
        this.refresh();
        // The document exists now; the agents only ever add to it.
        this.stream.start(ingested, ['about', 'prose']);
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
