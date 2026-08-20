import { Component, ElementRef, effect, inject, input, signal, viewChild } from '@angular/core';

import { ChatStream } from '../../../shared/chat-stream';
import { ChatTurn } from './chat-turn';

/**
 * The docked chat panel (R9.6).
 *
 * Autoscroll only while the reader is already at the bottom: scrolling up to
 * read an earlier tool result must not be yanked back down by the next delta.
 */
@Component({
  selector: 'app-chat-panel',
  imports: [ChatTurn],
  template: `
    <header class="head">
      <h2>Ask</h2>
      @if (chat.turns().length && !chat.streaming()) {
        <button class="clear" (click)="chat.reset()">clear</button>
      }
    </header>

    <div class="log" #log (scroll)="onScroll()">
      @for (turn of chat.turns(); track $index) {
        <app-chat-turn [turn]="turn" />
      } @empty {
        <p class="hint">
          Ask about this model — its quantization, what it needs to serve, how it compares to others
          in the vault.
        </p>
      }

      @if (chat.error(); as message) {
        <p class="error" role="alert">{{ message }}</p>
      }
    </div>

    <form class="composer" (submit)="send($event)">
      <textarea
        rows="2"
        [value]="draft()"
        (input)="draft.set($any($event.target).value)"
        (keydown.enter)="onEnter($event)"
        [disabled]="chat.streaming()"
        placeholder="Ask something…"
        aria-label="ask about this model"
      ></textarea>
      <button type="submit" [disabled]="chat.streaming() || !draft().trim()" aria-label="send">
        ↑
      </button>
    </form>
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      background: var(--sheet);
      color: var(--sheet-fg);
      border-radius: var(--radius);
      min-height: 0;
      overflow: hidden;
    }
    .head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: var(--space-4) var(--space-4) var(--space-2);
      border-bottom: 1px solid var(--sheet-border);
    }
    .head h2 {
      margin: 0;
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--sheet-faint);
    }
    .clear {
      background: none;
      border: none;
      color: var(--sheet-faint);
      font-size: 0.75rem;
      padding: 0;
    }
    .clear:hover {
      color: var(--sheet-fg);
    }
    .log {
      flex: 1;
      overflow-y: auto;
      padding: var(--space-4);
      min-height: 0;
    }
    .hint,
    .error {
      font-size: 0.8rem;
      line-height: 1.6;
    }
    .hint {
      color: var(--sheet-faint);
    }
    .error {
      color: #d33;
    }
    .composer {
      display: flex;
      gap: var(--space-2);
      padding: var(--space-3) var(--space-4) var(--space-4);
      border-top: 1px solid var(--sheet-border);
    }
    .composer textarea {
      flex: 1;
      resize: none;
      font: inherit;
      font-size: 0.85rem;
      padding: var(--space-2);
      border: 1px solid var(--sheet-border);
      border-radius: var(--radius-sm);
      background: transparent;
      color: inherit;
    }
    .composer button {
      align-self: flex-end;
      padding: var(--space-2) var(--space-3);
    }
  `,
})
export class ChatPanel {
  protected readonly chat = inject(ChatStream);

  readonly modelId = input.required<string>();

  protected readonly draft = signal('');
  private readonly log = viewChild.required<ElementRef<HTMLElement>>('log');
  private pinned = true;

  constructor() {
    // Re-reading turns() is what subscribes this effect to every delta.
    effect(() => {
      this.chat.turns();
      if (!this.pinned) return;
      queueMicrotask(() => {
        const el = this.log().nativeElement;
        el.scrollTop = el.scrollHeight;
      });
    });
  }

  protected onScroll(): void {
    const el = this.log().nativeElement;
    this.pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  protected onEnter(event: Event): void {
    // Shift+Enter is a newline; Enter sends. The composer is a textarea so a
    // long question can be written before it is asked.
    if ((event as KeyboardEvent).shiftKey) return;
    event.preventDefault();
    this.submit();
  }

  protected send(event: Event): void {
    event.preventDefault();
    this.submit();
  }

  private submit(): void {
    const message = this.draft().trim();
    if (!message || this.chat.streaming()) return;
    this.draft.set('');
    this.pinned = true;
    void this.chat.send(this.modelId(), message);
  }
}
