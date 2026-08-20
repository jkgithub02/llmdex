import { Component, computed, input } from '@angular/core';

import type { Turn } from '../../../shared/chat-stream';
import { Markdown } from '../../../shared/markdown';
import { ThinkingBlock } from '../../../shared/thinking-block';
import { ThinkingDots } from '../../../shared/thinking-dots';
import { ToolCall } from '../../../shared/tool-call';

/**
 * One turn, rendering its parts in the order they arrived (R9.10).
 *
 * A flat list rather than a thinking section, a tools section and an answer
 * section: a run interleaves them -- think, call two tools, answer -- and
 * grouping would misrepresent what the model did.
 */
@Component({
  selector: 'app-chat-turn',
  imports: [ThinkingBlock, ThinkingDots, ToolCall, Markdown],
  template: `
    <div class="turn" [attr.data-role]="turn().role">
      @for (part of turn().parts; track part.id) {
        @switch (part.kind) {
          @case ('thinking') {
            <app-thinking-block [text]="part.text" [done]="part.done" />
          }
          @case ('tool') {
            <app-tool-call
              [name]="part.name"
              [args]="part.args"
              [result]="part.result"
              [state]="part.state"
            />
          }
          @case ('text') {
            @if (turn().role === 'user') {
              <p class="text">{{ part.text }}</p>
            } @else {
              <app-markdown [text]="part.text" />
            }
          }
        }
      }

      @if (waiting()) {
        <app-thinking-dots />
      }
    </div>
  `,
  styles: `
    .turn {
      margin-bottom: var(--space-4);
    }
    .turn[data-role='user'] {
      background: var(--sheet-sunken);
      border-radius: var(--radius);
      padding: var(--space-3) var(--space-4);
    }
    .text {
      margin: 0 0 var(--space-2);
      font-size: 0.88rem;
      line-height: 1.6;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
  `,
})
export class ChatTurn {
  readonly turn = input.required<Turn>();
  /** Whether this turn is the one currently being generated. */
  readonly streaming = input(false);

  /**
   * Show the dots while the model owes us something visible.
   *
   * Not simply "streaming": once answer text is arriving, the text itself is
   * the progress indicator and a second one below it is noise. A tool still
   * running counts as waiting -- that is the longest silence in a run.
   */
  protected readonly waiting = computed(() => {
    if (!this.streaming()) return false;
    const parts = this.turn().parts;
    const last = parts[parts.length - 1];
    if (!last) return true;
    if (last.kind === 'text') return last.text === '';
    if (last.kind === 'tool') return last.state === 'running';
    return last.kind === 'thinking' ? last.done : true;
  });
}
