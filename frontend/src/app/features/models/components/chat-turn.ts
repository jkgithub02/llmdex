import { Component, input } from '@angular/core';

import type { Turn } from '../../../shared/chat-stream';
import { ThinkingBlock } from '../../../shared/thinking-block';
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
  imports: [ThinkingBlock, ToolCall],
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
            <p class="text">{{ part.text }}</p>
          }
        }
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
}
