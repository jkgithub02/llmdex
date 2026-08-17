import { Component, computed, input } from '@angular/core';

import { FieldState, STATE } from './field-state';

/**
 * The provenance badge. It is the smallest component here and the one the whole
 * product rests on: a reader must be able to tell at a glance whether a number
 * was computed, quoted, missing, or simply never measured.
 *
 * Colour is never the only signal — each state carries a distinct glyph and the
 * word itself, so the meaning survives greyscale and colour blindness.
 */
@Component({
  selector: 'app-state-badge',
  template: `
    <span class="badge" [class]="state()" [title]="hint()">
      <span class="glyph" aria-hidden="true">{{ style().glyph }}</span>
      {{ style().label }}
    </span>
  `,
  styles: `
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      font-size: 0.68rem;
      font-weight: 500;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 0.1rem 0.4rem;
      border-radius: 999px;
      border: 1px solid currentColor;
      background: color-mix(in srgb, currentColor 12%, transparent);
      white-space: nowrap;
      cursor: help;
    }
    .glyph {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      line-height: 1;
      opacity: 0.9;
    }
    .derived {
      color: var(--state-derived);
    }
    .extracted {
      color: var(--state-extracted);
    }
    .manual {
      color: var(--state-manual);
    }
    .absent {
      color: var(--state-absent);
    }
    .unmeasured {
      color: var(--state-unmeasured);
    }
  `,
})
export class StateBadge {
  readonly state = input.required<FieldState>();
  protected readonly style = computed(() => STATE[this.state()]);
  protected readonly hint = computed(() => STATE[this.state()].hint);
}
