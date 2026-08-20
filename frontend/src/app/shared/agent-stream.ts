/**
 * Reads an agent run as it happens.
 *
 * The reducer is static and pure so the rules can be tested without a browser
 * EventSource; the instance side is only the connection and the signal holding
 * its state. Nothing here is persisted -- the trace exists while you watch it
 * and then it is gone, which is why the document, not this, is the record.
 */

import { Injectable, NgZone, inject, signal } from '@angular/core';

export interface AgentState {
  name: string;
  phase: string;
  reasoning: string;
  error: string | null;
}

interface AgentPayload {
  agent?: string;
  phase?: string;
  text?: string;
  detail?: string;
}

@Injectable({ providedIn: 'root' })
export class AgentStream {
  private readonly zone = inject(NgZone);
  private source: EventSource | null = null;

  private readonly state = signal<AgentState[]>([]);
  private readonly active = signal(false);
  private readonly subject = signal<string | null>(null);

  readonly agents = this.state.asReadonly();
  readonly running = this.active.asReadonly();
  /** Which model this run belongs to, so a page can refuse to show another's. */
  readonly modelId = this.subject.asReadonly();

  /**
   * Fold one event into the agent list, without mutating what it was given --
   * the signal holds that array, and mutating in place would skip change
   * detection. The run's closing frame names no agent and is ignored here.
   */
  static applyEvent(states: AgentState[], kind: string, payload: AgentPayload): AgentState[] {
    const name = payload.agent;
    if (!name) return states;

    const existing = states.find((s) => s.name === name);
    const base: AgentState = existing ?? { name, phase: '', reasoning: '', error: null };
    const next: AgentState = {
      ...base,
      phase: kind === 'phase' ? (payload.phase ?? base.phase) : base.phase,
      reasoning: kind === 'reasoning' ? base.reasoning + (payload.text ?? '') : base.reasoning,
      error: kind === 'error' ? (payload.detail ?? 'failed') : base.error,
    };
    return existing ? states.map((s) => (s.name === name ? next : s)) : [...states, next];
  }

  /**
   * Follow a run already in flight, without starting one.
   *
   * Ingest returns the card and leaves the agents running, so a page opened
   * straight after lands mid-run. `attach` is the difference between showing
   * that run and paying for a second one -- and on a card whose blocks are
   * absent because an agent failed last week, between showing nothing and
   * silently spending tokens.
   */
  attach(modelId: string): void {
    this.start(modelId, ['about', 'prose', 'benchmarks'], true);
  }

  start(modelId: string, agents: string[], attach = false): void {
    this.stop();
    this.state.set([]);
    this.subject.set(modelId);
    this.active.set(true);

    const url =
      `/api/models/${modelId}/agents/stream?agents=${agents.join(',')}` +
      (attach ? '&attach=true' : '');
    const source = new EventSource(url);
    this.source = source;

    // EventSource fires outside Angular's zone, so every update is wrapped to
    // keep change detection running while the stream is open.
    const handle = (kind: string) => (event: MessageEvent) =>
      this.zone.run(() => {
        const payload = JSON.parse(event.data) as AgentPayload;
        if (kind === 'phase' && payload.phase === 'finished') {
          this.stop();
          return;
        }
        this.state.update((states) => AgentStream.applyEvent(states, kind, payload));
      });

    for (const kind of ['phase', 'reasoning', 'error']) {
      source.addEventListener(kind, handle(kind));
    }
    // A dropped stream is over. Without this EventSource reconnects on its own
    // and replays the whole run against agents that have already finished.
    source.onerror = () => this.zone.run(() => this.stop());
  }

  stop(): void {
    this.source?.close();
    this.source = null;
    this.active.set(false);
  }
}
