import { Component, computed, effect, input, signal } from '@angular/core';

/**
 * The model's chain of thought (R9.8).
 *
 * Expanded while it streams, so you watch it think; collapsed to one line when
 * it finishes, because by then the answer is what you want. Distinct from the
 * answer and never folded into it -- they arrive as separate parts and stay
 * separate on screen.
 *
 * The auto-collapse yields to a manual one. Nothing is more irritating than a
 * panel that closes what you just opened.
 */
@Component({
  selector: 'app-thinking-block',
  template: `
    <button class="strip" (click)="toggle()" [attr.aria-expanded]="open()">
      <span class="chevron" [class.down]="open()">▸</span>
      <span class="label">{{ label() }}</span>
    </button>

    @if (open()) {
      <pre class="cot">{{ text() || '…' }}</pre>
    }
  `,
  styles: `
    :host {
      display: block;
      margin: var(--space-2) 0;
    }
    .strip {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      background: none;
      border: none;
      border-radius: 0;
      padding: 0;
      color: var(--sheet-faint);
      font-size: 0.78rem;
      font-weight: 500;
    }
    .strip:hover {
      color: var(--sheet-fg);
    }
    .chevron {
      display: inline-block;
      transition: transform 120ms;
      font-size: 0.7rem;
    }
    .chevron.down {
      transform: rotate(90deg);
    }
    @media (prefers-reduced-motion: reduce) {
      .chevron {
        transition: none;
      }
    }
    .cot {
      margin: var(--space-2) 0 0;
      padding-left: var(--space-3);
      border-left: 2px solid var(--sheet-border);
      color: var(--sheet-faint);
      font-size: 0.78rem;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 18rem;
      overflow-y: auto;
    }
  `,
})
export class ThinkingBlock {
  readonly text = input.required<string>();
  readonly done = input.required<boolean>();

  protected readonly open = signal(true);
  private readonly touched = signal(false);
  private readonly startedAt = Date.now();
  private readonly elapsed = signal<number | null>(null);

  protected readonly label = computed(() => {
    if (!this.done()) return 'Thinking…';
    const seconds = this.elapsed();
    return seconds === null ? 'Thought' : `Thought for ${seconds}s`;
  });

  constructor() {
    effect(() => {
      if (!this.done()) return;
      if (this.elapsed() === null) {
        this.elapsed.set(Math.max(1, Math.round((Date.now() - this.startedAt) / 1000)));
      }
      // Auto-collapse, unless the reader has already made the call themselves.
      if (!this.touched()) this.open.set(false);
    });
  }

  protected toggle(): void {
    this.touched.set(true);
    this.open.update((v) => !v);
  }
}
