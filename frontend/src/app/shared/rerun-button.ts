import { Component, computed, inject, input, output } from '@angular/core';

import { AgentStream } from './agent-stream';

/**
 * Re-run one agent, and show that it is running.
 *
 * A disabled button says "not now"; it does not say "working". These runs take
 * tens of seconds against a real endpoint, and the only other signal was a
 * trace panel elsewhere on the page -- so the tab you pressed the button on
 * looked like it had ignored you.
 *
 * The running state is read from AgentStream rather than passed in: the stream
 * already knows which agent is live, and threading a second boolean through
 * four tabs to say the same thing is how the two drift apart.
 */
@Component({
  selector: 'app-rerun-button',
  template: `
    <button class="ghost" (click)="rerun.emit(agent())" [disabled]="disabled()">
      @if (running()) {
        <span class="sweep" aria-hidden="true"></span>
        <span class="label">{{ phase() }}</span>
      } @else {
        <span class="label">{{ label() }}</span>
      }
    </button>
  `,
  styles: `
    button {
      position: relative;
      overflow: hidden;
    }
    .label {
      position: relative;
      z-index: 1;
    }
    /* Back and forth rather than left-to-right: a run has no percentage to
       report, and a bar that fills implies one. */
    .sweep {
      position: absolute;
      inset: 0;
      background: linear-gradient(
        90deg,
        transparent 0%,
        color-mix(in srgb, var(--accent, #3f6ff5) 26%, transparent) 50%,
        transparent 100%
      );
      width: 60%;
      animation: sweep 1.1s ease-in-out infinite alternate;
    }
    @keyframes sweep {
      from {
        transform: translateX(-70%);
      }
      to {
        transform: translateX(170%);
      }
    }
    @media (prefers-reduced-motion: reduce) {
      .sweep {
        animation: none;
        width: 100%;
        opacity: 0.5;
      }
    }
  `,
})
export class RerunButton {
  private readonly stream = inject(AgentStream);

  /** The agent's name on the wire: about | spec | prose | benchmarks. */
  readonly agent = input.required<string>();
  readonly label = input('Re-run');
  /** Another agent is busy, so this one may not start. */
  readonly busy = input(false);
  readonly rerun = output<string>();

  protected readonly running = computed(() => {
    if (!this.stream.running()) return false;
    const mine = this.stream.agents().find((a) => a.name === this.agent());
    return !!mine && mine.phase !== 'done' && !mine.error;
  });

  protected readonly disabled = computed(() => this.busy() || this.running());

  /** "started" is what the wire says; "Running…" is what a reader wants. */
  protected readonly phase = computed(() => {
    const mine = this.stream.agents().find((a) => a.name === this.agent());
    return mine?.phase === 'started' || !mine?.phase ? 'Running…' : `${mine.phase}…`;
  });
}
