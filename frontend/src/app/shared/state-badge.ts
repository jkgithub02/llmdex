import { Component, computed, input } from '@angular/core';

import { FieldState, STATE } from './provenance';

/**
 * The provenance badge. It is the smallest component here and the one the whole
 * product rests on: a reader must be able to tell at a glance whether a number
 * was computed, quoted, missing, or simply never measured.
 *
 * Colour is never the only signal — the word is always there, so the meaning
 * survives greyscale and colour blindness. The colour is carried by a dot
 * rather than by the label, which keeps the label legible on both the dark grid
 * and the light sheet: it is mixed from whatever colour it inherits, so one
 * rule covers every surface this appears on.
 */
@Component({
  selector: 'app-state-badge',
  template: `
    <span class="badge" [class]="state()" [title]="hint()">
      <span class="dot" aria-hidden="true"></span>{{ style().label }}
    </span>
  `,
  styles: `
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.65rem;
      font-weight: 500;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      /* Quieter than the value it annotates. A hundred outlined pills shouting
         at the same weight as the data is what made this look like a toy. */
      color: color-mix(in srgb, currentColor 60%, transparent);
      white-space: nowrap;
      cursor: help;
    }
    .dot {
      width: 0.4rem;
      height: 0.4rem;
      flex: none;
      border-radius: 999px;
      background: var(--dot);
    }
    .derived {
      --dot: var(--state-derived);
    }
    .extracted {
      --dot: var(--state-extracted);
    }
    .manual {
      --dot: var(--state-manual);
    }
    .absent {
      --dot: var(--state-absent);
    }
    .unmeasured {
      --dot: var(--state-unmeasured);
    }
  `,
})
export class StateBadge {
  readonly state = input.required<FieldState>();
  protected readonly style = computed(() => STATE[this.state()]);
  protected readonly hint = computed(() => STATE[this.state()].hint);
}
