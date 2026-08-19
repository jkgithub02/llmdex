import { Component, input } from '@angular/core';

import { StateBadge } from '../../../shared/state-badge';
import type { ModelDoc } from '../../../api/model/modelDoc';
import { derivedFields } from '../fields';

/** The Spec tab: everything computed from `config.json` — never guessed (R2.x). */
@Component({
  selector: 'app-spec-tab',
  imports: [StateBadge],
  template: `
    @for (
      checkpoint of model().checkpoints ?? [];
      track checkpoint.repo + checkpoint.quantization
    ) {
      <section class="pane">
        <div class="ck-head">
          <span class="mono repo">{{ checkpoint.repo }}</span>
          @if (checkpoint.quantization) {
            <span class="tag mono">{{ checkpoint.quantization }}</span>
          }
          <span class="rev mono" title="the card revision this was built from (R1.3)">
            &#64;{{ checkpoint.card_revision?.slice(0, 7) ?? 'unknown' }}
          </span>
        </div>
        <p class="hint">computed from config.json — never guessed</p>

        <div class="grid">
          @for (field of derivedFields(checkpoint); track field.label) {
            <div class="field">
              <span class="key">{{ field.label }}</span>
              <span class="val mono" [class.null]="field.value === null">
                {{ field.value ?? 'unavailable' }}
              </span>
              <app-state-badge [state]="field.state" />
              @if (field.unreliable) {
                <p class="why">{{ field.unreliable }}</p>
              }
            </div>
          }
        </div>
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

    .ck-head {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      flex-wrap: wrap;
    }
    .repo {
      font-size: 0.9rem;
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    .tag {
      font-size: 0.7rem;
      padding: 0.1rem 0.4rem;
      border-radius: var(--radius-sm);
      background: var(--sheet-sunken);
      border: 1px solid var(--sheet-border);
    }
    .rev {
      margin-left: auto;
      font-size: 0.75rem;
      color: var(--sheet-faint);
      cursor: help;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr));
      gap: var(--space-2);
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
    .hint {
      margin: 0 0 var(--space-3);
      font-size: 0.78rem;
      color: var(--sheet-muted);
    }
  `,
})
export class SpecTab {
  readonly model = input.required<ModelDoc>();

  protected readonly derivedFields = derivedFields;
}
