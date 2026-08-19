import { Component, computed, input } from '@angular/core';

import type { LayerComposition } from '../api/model/layerComposition';
import { layerSegments } from './architecture';

/**
 * The layer stack as proportional bands.
 *
 * This is the detail page's headline graphic, standing where a Pokedex puts the
 * creature. It earns that place by being real: the bands are the composition
 * `config.json` declared (R2.6a), so a dense transformer reads as one solid bar
 * and a hybrid reads as a striped one you can recognise across the vault.
 *
 * Proportional bands rather than one tick per layer, deliberately.
 * `LayerComposition` stores counts, not the order they appear in, and drawing 52
 * ticks would show a sequence nobody read. Widths are a claim we can support;
 * position is not.
 */
@Component({
  selector: 'app-layer-strip',
  template: `
    @if (segments().length) {
      <p class="caption">layer composition · {{ total() }} layers</p>
      <div class="strip" [attr.aria-label]="summary()">
        @for (segment of segments(); track segment.kind) {
          <span
            class="band"
            [class]="'band-' + segment.kind"
            [style.flex-grow]="segment.count"
            [style.background-size]="'calc(100% / ' + segment.count + ') 100%'"
            [title]="segment.label"
          ></span>
        }
      </div>
      <ul class="legend">
        @for (segment of segments(); track segment.kind) {
          <li><i [class]="'dot dot-' + segment.kind"></i>{{ segment.label }}</li>
        }
      </ul>
    } @else {
      <p class="none">layer composition unavailable</p>
    }
  `,
  styles: `
    :host {
      display: block;
    }
    .strip {
      display: flex;
      gap: 3px;
      height: 3.25rem;
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .band {
      border-radius: 3px;
      /* One tick per layer within the band. Ticks divide layers of the same
         kind, so they show how many without implying where in the stack they
         sit -- an order LayerComposition does not record. */
      background-image: linear-gradient(90deg, transparent calc(100% - 1px), rgb(0 0 0 / 0.22) 0);
      transition: opacity 180ms ease;
    }
    /* Opacity rather than hue: the strip sits on the tone colour of whichever
       architecture it belongs to, and must stay legible on all of them. */
    /* background-color rather than the shorthand throughout: the shorthand
       would reset the tick gradient set above. */
    .band-attention {
      background-color: rgb(255 255 255 / 0.95);
    }
    .band-recurrent {
      background-color: rgb(255 255 255 / 0.55);
    }
    .band-mlp {
      background-color: rgb(255 255 255 / 0.25);
    }
    .caption {
      margin: 0 0 var(--space-2);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgb(255 255 255 / 0.75);
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2) var(--space-4);
      list-style: none;
      margin: var(--space-3) 0 0;
      padding: 0;
      font-size: 0.75rem;
      color: rgb(255 255 255 / 0.85);
    }
    .legend li {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .dot {
      width: 0.6rem;
      height: 0.6rem;
      border-radius: 2px;
    }
    .dot-attention {
      background: rgb(255 255 255 / 0.95);
    }
    .dot-recurrent {
      background: rgb(255 255 255 / 0.55);
    }
    .dot-mlp {
      background: rgb(255 255 255 / 0.25);
    }
    .none {
      margin: 0;
      font-size: 0.8rem;
      font-style: italic;
      color: rgb(255 255 255 / 0.7);
    }
  `,
})
export class LayerStrip {
  readonly layers = input<LayerComposition | null | undefined>();

  protected readonly segments = computed(() => layerSegments(this.layers()));
  protected readonly total = computed(() =>
    this.segments().reduce((sum, segment) => sum + segment.count, 0),
  );
  protected readonly summary = computed(() =>
    this.segments()
      .map((s) => s.label)
      .join(', '),
  );
}
