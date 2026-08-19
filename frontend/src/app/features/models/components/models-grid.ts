import { Component, input, output } from '@angular/core';
import { RouterLink } from '@angular/router';

import type { ModelDoc } from '../../../api/model/modelDoc';
import { formatBytes, formatCount, routeFor } from '../../../shared/format';
import { StateBadge } from '../../../shared/state-badge';
import { architecturePills, architectureTone } from '../../../shared/architecture';
import { LayerStrip } from '../../../shared/layer-strip';

/**
 * The card-grid view: one Pokedex-style card per model.
 * Presentation only — the confirm/delete state lives on the page (shared with
 * the table) and is passed down, so `remove` fires the actual delete and
 * `confirmChange` reports the reader asking to confirm or backing out.
 */
@Component({
  selector: 'app-models-grid',
  imports: [RouterLink, StateBadge, LayerStrip],
  template: `
    <ul class="dex-grid">
      @for (row of rows(); track row.model_id) {
        <li>
          <!-- Outside the anchor on purpose: a button inside a link is
             invalid HTML, and the click would navigate before deleting. -->
          @if (confirming() === row.model_id) {
            <div class="confirm">
              <span>Delete?</span>
              <button type="button" class="yes" (click)="remove.emit(row.model_id)">
                {{ deleting() === row.model_id ? '…' : 'Delete' }}
              </button>
              <button type="button" class="no" (click)="confirmChange.emit(null)">Keep</button>
            </div>
          } @else {
            <button
              type="button"
              class="del"
              [attr.aria-label]="'delete ' + row.model_id"
              (click)="confirmChange.emit(row.model_id)"
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
                <dd class="mono" [class.withheld]="!vram(row)" [title]="vramReason(row) ?? ''">
                  {{ vram(row) ?? 'withheld' }}
                </dd>
              </div>
            </dl>
          </a>
        </li>
      }
    </ul>
  `,
  styles: `
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
export class ModelsGrid {
  readonly rows = input.required<ModelDoc[]>();
  readonly confirming = input.required<string | null>();
  readonly deleting = input.required<string | null>();
  readonly remove = output<string>();
  readonly confirmChange = output<string | null>();

  protected route = routeFor;

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
