import {
  Component,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  viewChildren,
} from '@angular/core';

import { AgentStream } from './agent-stream';

/**
 * What the agents are thinking, while they think it.
 *
 * The reasoning is shown and then thrown away -- it is not stored on the
 * document, so this panel is the only place it ever exists. Each agent gets its
 * own pane because they run concurrently and their output interleaves, and each
 * pane lives in the tab that shows what that agent wrote.
 */
@Component({
  selector: 'app-agent-trace',
  template: `
    @if (mine() && panes().length) {
      <section class="trace">
        @for (agent of panes(); track agent.name) {
          <article class="pane" [class.failed]="agent.error">
            <header>
              <span class="name">{{ agent.name }}</span>
              @if (agent.error) {
                <span class="phase err">{{ agent.error }}</span>
              } @else {
                <span class="phase">{{ agent.phase }}</span>
              }
            </header>
            @if (agent.reasoning) {
              <pre #scroller>{{ agent.reasoning }}</pre>
            }
          </article>
        }
      </section>
    }
  `,
  styles: `
    .trace {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr));
      gap: var(--space-3);
      margin-bottom: var(--space-4);
    }
    .pane {
      background: var(--bg-sunken);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: var(--space-3);
    }
    .pane.failed {
      border-color: var(--danger);
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: var(--space-2);
    }
    .name {
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--fg);
    }
    .phase {
      font-size: 0.72rem;
      color: var(--fg-faint);
    }
    .phase.err {
      color: var(--danger);
    }
    pre {
      margin: var(--space-2) 0 0;
      /* Capped: a trace runs to thousands of tokens and must not grow the page. */
      max-height: 9rem;
      overflow-y: auto;
      white-space: pre-wrap;
      font-size: 0.72rem;
      line-height: 1.45;
      color: var(--fg-muted);
    }
  `,
})
export class AgentTrace {
  protected readonly stream = inject(AgentStream);

  /**
   * Show only a run belonging to this model. The stream is a root singleton, so
   * without this a detail page renders whatever run is in flight -- ingest model
   * A, open model B, and B shows A's reasoning as if it were its own. Unset
   * means "whatever is running", which is what the models list wants.
   */
  readonly only = input<string | null>(null);

  /**
   * Show only this agent. A tab shows what one agent wrote, so it shows that
   * agent's thinking and not its neighbour's. Unset means every agent running.
   */
  readonly agent = input<string | null>(null);

  protected readonly mine = computed(() => {
    const only = this.only();
    return !only || this.stream.modelId() === only;
  });

  protected readonly panes = computed(() => {
    const wanted = this.agent();
    const agents = this.stream.agents();
    return wanted ? agents.filter((a) => a.name === wanted) : agents;
  });
  private readonly scrollers = viewChildren<ElementRef<HTMLElement>>('scroller');

  constructor() {
    // Follow the newest text, the way a log viewer does.
    effect(() => {
      this.stream.agents();
      for (const scroller of this.scrollers()) {
        scroller.nativeElement.scrollTop = scroller.nativeElement.scrollHeight;
      }
    });
  }
}
