import { Component } from '@angular/core';

/**
 * The gap between asking and the first frame.
 *
 * On an endpoint that streams reasoning, that gap is filled by the thinking
 * block. On one that does not -- Gemma emits no reasoning parts at all -- the
 * panel would otherwise sit blank through a tool call and a second model
 * request, which reads as nothing happening.
 */
@Component({
  selector: 'app-thinking-dots',
  template: `
    <span class="dots" role="status" aria-label="waiting for a response">
      <span></span><span></span><span></span>
    </span>
  `,
  styles: `
    :host {
      display: block;
      padding: var(--space-2) 0;
    }
    .dots {
      display: inline-flex;
      gap: 0.28rem;
      align-items: center;
    }
    .dots span {
      width: 0.35rem;
      height: 0.35rem;
      border-radius: 999px;
      background: var(--sheet-faint);
      animation: blink 1.2s ease-in-out infinite;
    }
    .dots span:nth-child(2) {
      animation-delay: 0.2s;
    }
    .dots span:nth-child(3) {
      animation-delay: 0.4s;
    }
    @keyframes blink {
      0%,
      80%,
      100% {
        opacity: 0.25;
      }
      40% {
        opacity: 1;
      }
    }
    /* Still visible, just still: the indicator says "working", and removing it
       entirely would say "idle". */
    @media (prefers-reduced-motion: reduce) {
      .dots span {
        animation: none;
        opacity: 0.6;
      }
    }
  `,
})
export class ThinkingDots {}
