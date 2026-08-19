import { Component, input, output } from '@angular/core';

import { AgentTrace } from '../../../shared/agent-trace';
import { StateBadge } from '../../../shared/state-badge';
import type { ModelDoc } from '../../../api/model/modelDoc';
import { extractedFields } from '../fields';

/** The Details tab: every extracted field, copied from the card — never generated (R3.x). */
@Component({
  selector: 'app-details-tab',
  imports: [AgentTrace, StateBadge],
  template: `
    <app-agent-trace [only]="modelId()" agent="prose" />
    @for (
      checkpoint of model().checkpoints ?? [];
      track checkpoint.repo + checkpoint.quantization
    ) {
      <section class="pane">
        <p class="hint">copied from the card — never generated</p>
        @if (checkpoint.extracted; as extracted) {
          <p class="prov">
            extracted by <code>{{ extracted.model }}</code> on {{ extracted.extracted_on }} from
            card
            <code>{{ extracted.card_revision.slice(0, 7) }}</code>
          </p>
          <button class="ghost" (click)="rerun.emit('prose')" [disabled]="busy()">
            {{ extracting() ? 'Reading the card…' : 'Re-run' }}
          </button>
          <div class="grid">
            @for (field of extractedFields(checkpoint); track field.label) {
              <div class="field">
                <span class="key">{{ field.label }}</span>
                <span class="val" [class.null]="field.value === null">
                  {{ field.value ?? 'unavailable' }}
                </span>
                <app-state-badge [state]="field.state" />
                @if (field.source) {
                  <p class="why">§ {{ field.source }}</p>
                }
              </div>
            }
          </div>
        } @else {
          <div class="cta">
            <p>Nobody has read this card yet.</p>
            <button (click)="rerun.emit('prose')" [disabled]="busy()">
              {{ extracting() ? 'Reading the card…' : 'Extract with the LLM' }}
            </button>
          </div>
          @if (extracting()) {
            <div class="bar"><span></span></div>
            <p class="hint">
              The whole card goes to the model, which returns quotes. Only quotes we can find in the
              card are kept. About a minute.
            </p>
          }
        }
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
    .hint,
    .prov {
      font-size: 0.78rem;
      color: var(--sheet-muted);
    }
    .hint {
      margin: 0 0 var(--space-3);
    }
    .prov {
      margin: var(--space-4) 0 var(--space-3);
    }
    code {
      background: var(--sheet-sunken);
      padding: 0.05rem 0.3rem;
      border-radius: 4px;
    }
    .cta {
      display: flex;
      gap: var(--space-4);
      align-items: center;
      flex-wrap: wrap;
      padding: var(--space-4);
      border: 1px dashed var(--sheet-border);
      border-radius: var(--radius-sm);
      color: var(--sheet-muted);
    }
    .pane button {
      background: var(--tone);
      color: #fff;
      border: none;
    }
    .pane button:hover:not(:disabled) {
      filter: brightness(1.08);
    }
    button.ghost {
      margin-top: var(--space-4);
      background: none;
      border: 1px solid var(--sheet-border);
      color: var(--sheet-muted);
      font-size: 0.78rem;
      padding: 0.3rem 0.7rem;
    }
    button.ghost:hover:not(:disabled) {
      filter: none;
      color: var(--sheet-fg);
      border-color: var(--sheet-faint);
    }
  `,
})
export class DetailsTab {
  readonly model = input.required<ModelDoc>();
  readonly modelId = input.required<string>();
  readonly busy = input.required<boolean>();
  readonly extracting = input.required<boolean>();
  readonly rerun = output<string>();

  protected readonly extractedFields = extractedFields;
}
