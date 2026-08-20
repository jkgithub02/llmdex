import {
  Component,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';

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
      <div class="who">
        <span class="vendor">{{ vendor() }}</span>
        <h2 [title]="modelId()">{{ shortName() }}</h2>
      </div>

      <div class="actions">
        @if (chat.turns().length && !chat.streaming()) {
          <button class="ghost" (click)="chat.reset()" title="clear this conversation">
            clear
          </button>
        }
        <button
          class="close"
          (click)="closed.emit()"
          aria-label="close the chat panel"
          title="close"
        >
          ✕
        </button>
      </div>
    </header>

    <div class="log" #log (scroll)="onScroll()">
      @for (turn of chat.turns(); track $index; let last = $last) {
        <app-chat-turn [turn]="turn" [streaming]="last && chat.streaming()" />
      } @empty {
        <div class="empty">
          <p class="hint">
            Grounded in this model's reviewed document and its card. It can read the rest of the
            vault, and search the web when neither covers the question.
          </p>
          <div class="suggestions">
            @for (s of suggestions; track s) {
              <button class="chip" (click)="ask(s)">{{ s }}</button>
            }
          </div>
        </div>
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
      align-items: flex-start;
      justify-content: space-between;
      gap: var(--space-3);
      padding: var(--space-3) var(--space-4);
      border-bottom: 1px solid var(--sheet-border);
      background: var(--sheet-sunken);
    }
    .who {
      min-width: 0;
    }
    .vendor {
      display: block;
      font-size: 0.65rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--sheet-faint);
    }
    /* The model, not the feature: you always know which card you are asking
       about, which matters once two tabs are open on two models. */
    .head h2 {
      margin: 0.1rem 0 0;
      font-size: 0.9rem;
      font-weight: 600;
      line-height: 1.25;
      color: var(--sheet-fg);
      overflow-wrap: anywhere;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      flex: none;
    }
    .ghost {
      background: none;
      border: none;
      color: var(--sheet-faint);
      font-size: 0.72rem;
      padding: 0;
    }
    .ghost:hover {
      color: var(--sheet-fg);
    }
    /* Deliberately a real target, not a 12px glyph: the edge handle alone was
       fiddly to hit. */
    .close {
      display: grid;
      place-items: center;
      width: 1.75rem;
      height: 1.75rem;
      padding: 0;
      border: 1px solid transparent;
      border-radius: var(--radius-sm);
      background: none;
      color: var(--sheet-faint);
      font-size: 0.8rem;
      line-height: 1;
    }
    .close:hover {
      background: var(--sheet);
      border-color: var(--sheet-border);
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
      margin: 0 0 var(--space-3);
    }
    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
    }
    .chip {
      background: var(--sheet-sunken);
      border: 1px solid var(--sheet-border);
      border-radius: 999px;
      padding: 0.3rem 0.7rem;
      font-size: 0.75rem;
      color: var(--sheet-muted);
      text-align: left;
    }
    .chip:hover {
      color: var(--sheet-fg);
      border-color: var(--sheet-faint);
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
  readonly closed = output<void>();

  /** `vendor/name` -- the header shows them apart. */
  protected readonly vendor = computed(() => this.modelId().split('/')[0] ?? '');
  protected readonly shortName = computed(() => this.modelId().split('/').slice(1).join('/'));

  protected readonly suggestions = [
    'What quantization does it use?',
    'What does it need to serve?',
    'How does it compare to the others in the vault?',
  ];

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

  /** A suggestion chip is just a question already typed. */
  protected ask(question: string): void {
    if (this.chat.streaming()) return;
    this.draft.set(question);
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
