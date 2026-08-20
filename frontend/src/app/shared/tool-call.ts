import { Component, computed, input, signal } from '@angular/core';

/**
 * One tool call (R9.9).
 *
 * Collapsed it is a line: the tool, its arguments, and a preview of what came
 * back. The leading dot carries state, so running / done / failed reads without
 * expanding -- an error you have to go looking for is the same defect as a
 * swallowed exception.
 *
 * Arguments render as `key: value`. The model sends
 * {"section": "Training Methodology"}; a reader wants
 * section: "Training Methodology".
 */
@Component({
  selector: 'app-tool-call',
  template: `
    <button class="line" (click)="open.set(!open())" [attr.aria-expanded]="open()">
      <span class="dot" [attr.data-state]="state()"></span>
      <span class="call mono">{{ name() }}({{ prettyArgs() }})</span>
    </button>

    @if (!open() && preview()) {
      <div class="preview mono" [class.failed]="state() === 'error'">⎿ {{ preview() }}</div>
    }

    @if (open()) {
      <div class="detail">
        <div class="heading">arguments</div>
        <pre class="mono">{{ fullArgs() }}</pre>
        @if (result(); as body) {
          <div class="heading">{{ state() === 'error' ? 'error' : 'result' }}</div>
          <pre class="mono" [class.failed]="state() === 'error'">{{ body }}</pre>
        }
      </div>
    }
  `,
  styles: `
    :host {
      display: block;
      margin: var(--space-2) 0;
    }
    .line {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      background: none;
      border: none;
      border-radius: 0;
      padding: 0;
      color: var(--sheet-fg);
      font-size: 0.78rem;
      text-align: left;
    }
    .dot {
      width: 0.5rem;
      height: 0.5rem;
      border-radius: 999px;
      flex: none;
      background: var(--sheet-faint);
    }
    /* Semantic rather than the architecture tone: this panel renders outside
       the .dex element that defines --tone, and "the call worked" is not a
       fact about the model's architecture. */
    .dot[data-state='running'] {
      background: var(--sheet-muted);
      animation: pulse 1s ease-in-out infinite;
    }
    .dot[data-state='done'] {
      background: var(--type-hybrid);
    }
    .dot[data-state='error'] {
      background: #d33;
    }
    @keyframes pulse {
      50% {
        opacity: 0.25;
      }
    }
    @media (prefers-reduced-motion: reduce) {
      .dot[data-state='running'] {
        animation: none;
      }
    }
    .call {
      overflow-wrap: anywhere;
    }
    .preview,
    .detail {
      padding-left: calc(0.5rem + var(--space-2));
      color: var(--sheet-faint);
      font-size: 0.75rem;
    }
    .preview {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .failed {
      color: #d33;
    }
    .heading {
      margin-top: var(--space-2);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.65rem;
    }
    .detail pre {
      margin: var(--space-1) 0 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 20rem;
      overflow-y: auto;
    }
  `,
})
export class ToolCall {
  readonly name = input.required<string>();
  readonly args = input.required<string>();
  readonly result = input.required<string | null>();
  readonly state = input.required<'running' | 'done' | 'error'>();

  protected readonly open = signal(false);

  private readonly parsed = computed<Record<string, unknown> | null>(() => {
    try {
      const value = JSON.parse(this.args() || '{}');
      return value && typeof value === 'object' ? value : null;
    } catch {
      // A tool call whose arguments are not JSON is shown raw rather than
      // hidden: it is exactly the case someone will need to see.
      return null;
    }
  });

  /** One line: `section: "Training Methodology"`, truncated. */
  protected readonly prettyArgs = computed(() => {
    const value = this.parsed();
    if (!value) return this.args();
    const rendered = Object.entries(value)
      .map(([key, v]) => `${key}: ${JSON.stringify(v)}`)
      .join(', ');
    return rendered.length > 80 ? `${rendered.slice(0, 79)}…` : rendered;
  });

  protected readonly fullArgs = computed(() => {
    const value = this.parsed();
    return value ? JSON.stringify(value, null, 2) : this.args() || '(none)';
  });

  /**
   * First line of the result, plus how much more there is.
   *
   * A bare first line misreports a multi-line answer: `list_models` returning
   * four models showed one truncated row, which reads as the tool having found
   * one. The count is what tells you to expand.
   */
  protected readonly preview = computed(() => {
    const body = this.result();
    if (this.state() === 'error') return body ?? 'failed';
    if (!body) return '';
    const lines = body.split('\n').filter((l) => l.trim());
    const first = lines[0] ?? '';
    const head = first.length > 80 ? `${first.slice(0, 79)}…` : first;
    return lines.length > 1 ? `${head}  (+${lines.length - 1} more)` : head;
  });
}
