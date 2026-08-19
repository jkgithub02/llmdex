import { Component, input, output } from '@angular/core';
import { RouterLink } from '@angular/router';

import type { ModelDoc } from '../../../api/model/modelDoc';
import { routeFor } from '../../../shared/format';
import { tableRow } from '../table-row';

/**
 * The dense table view: one row per model, same facts as the grid's card.
 * Presentation only — the confirm/delete state lives on the page (shared with
 * the grid) and is passed down, so `remove` fires the actual delete and
 * `confirmChange` reports the reader asking to confirm or backing out.
 */
@Component({
  selector: 'app-models-table',
  imports: [RouterLink],
  template: `
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
          @for (row of rows(); track row.model_id) {
            @let cells = tableRow(row);
            <tr>
              <th scope="row">
                <a [routerLink]="route(cells.modelId)">
                  <span class="t-vendor">{{ cells.vendor }}/</span
                  ><span class="mono">{{ cells.name }}</span>
                </a>
              </th>
              <td>
                <span class="chip" [attr.data-tone]="cells.tone">{{ cells.architecture }}</span>
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
                  <button type="button" class="yes" (click)="remove.emit(cells.modelId)">
                    {{ deleting() === cells.modelId ? '…' : 'Delete' }}
                  </button>
                  <button type="button" class="no" (click)="confirmChange.emit(null)">Keep</button>
                } @else {
                  <button
                    type="button"
                    class="del-inline"
                    [attr.aria-label]="'delete ' + cells.modelId"
                    (click)="confirmChange.emit(cells.modelId)"
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
  `,
  styles: `
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
  `,
})
export class ModelsTable {
  readonly rows = input.required<ModelDoc[]>();
  readonly confirming = input.required<string | null>();
  readonly deleting = input.required<string | null>();
  readonly remove = output<string>();
  readonly confirmChange = output<string | null>();

  protected route = routeFor;
  protected tableRow = tableRow;
}
