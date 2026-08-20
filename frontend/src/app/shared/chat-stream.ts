/**
 * Reads one chat turn as it happens.
 *
 * fetch() and a stream reader rather than EventSource: the transcript goes in
 * the request body and EventSource is GET-only.
 *
 * The reducer is static and pure so the rules can be tested without a browser,
 * the same shape as AgentStream. Arrival order is render order (R9.10) -- the
 * server assembles the part ids, because pydantic-ai's indices reset on every
 * model request and an index alone is not a key.
 */

import { Injectable, NgZone, inject, signal } from '@angular/core';

export type Part =
  | { kind: 'thinking'; id: string; text: string; done: boolean }
  | { kind: 'text'; id: string; text: string }
  | {
      kind: 'tool';
      id: string;
      name: string;
      args: string;
      result: string | null;
      state: 'running' | 'done' | 'error';
    };

export interface Turn {
  role: 'user' | 'assistant';
  parts: Part[];
}

interface Payload {
  part_id?: string;
  part?: string;
  text?: string;
  tool_call_id?: string;
  tool?: string;
  args?: string;
  result?: string;
  ok?: boolean;
  detail?: string;
  phase?: string;
  messages?: unknown[];
}

@Injectable({ providedIn: 'root' })
export class ChatStream {
  private readonly zone = inject(NgZone);
  private controller: AbortController | null = null;

  private readonly state = signal<Turn[]>([]);
  private readonly active = signal(false);
  private readonly failure = signal<string | null>(null);
  /** Opaque: stored and sent back, never interpreted here (R9.4). */
  private history: unknown[] = [];

  readonly turns = this.state.asReadonly();
  readonly streaming = this.active.asReadonly();
  readonly error = this.failure.asReadonly();

  /**
   * Fold one frame into the turn list, without mutating what it was given --
   * the signal holds that array, and mutating in place would skip change
   * detection. Everything lands on the last turn, which is the open one.
   */
  static applyEvent(turns: Turn[], kind: string, payload: Payload): Turn[] {
    const last = turns[turns.length - 1];
    if (!last || last.role !== 'assistant') return turns;

    const parts = ChatStream.applyToParts(last.parts, kind, payload);
    if (parts === last.parts) return turns;
    return [...turns.slice(0, -1), { ...last, parts }];
  }

  private static applyToParts(parts: Part[], kind: string, payload: Payload): Part[] {
    const replace = (id: string, next: Part): Part[] => parts.map((p) => (p.id === id ? next : p));

    switch (kind) {
      case 'part_start': {
        const id = payload.part_id ?? '';
        // A part can open with its first chunk already attached; the deltas
        // that follow carry only the rest.
        const opening = payload.text ?? '';
        if (payload.part === 'thinking') {
          return [...parts, { kind: 'thinking', id, text: opening, done: false }];
        }
        return [...parts, { kind: 'text', id, text: opening }];
      }

      case 'reasoning':
      case 'content': {
        const id = payload.part_id ?? '';
        const found = parts.find((p) => p.id === id);
        if (!found || (found.kind !== 'thinking' && found.kind !== 'text')) return parts;
        return replace(id, { ...found, text: found.text + (payload.text ?? '') });
      }

      case 'part_end': {
        const id = payload.part_id ?? '';
        const found = parts.find((p) => p.id === id);
        if (!found) return parts;
        // Measured: an empty TextPart opens and closes on every run. Rendering
        // it produces an empty bubble. An empty *thinking* part is kept -- it
        // still says the model thought, and dropping it would misreport the turn.
        if (found.kind === 'text' && found.text === '') {
          return parts.filter((p) => p.id !== id);
        }
        if (found.kind === 'thinking') return replace(id, { ...found, done: true });
        return parts;
      }

      case 'tool_call':
        return [
          ...parts,
          {
            kind: 'tool',
            id: payload.tool_call_id ?? '',
            name: payload.tool ?? '',
            args: payload.args ?? '',
            result: null,
            state: 'running',
          },
        ];

      case 'tool_result': {
        const id = payload.tool_call_id ?? '';
        const found = parts.find((p) => p.id === id);
        if (!found || found.kind !== 'tool') return parts;
        return replace(id, {
          ...found,
          result: payload.result ?? '',
          state: payload.ok === false ? 'error' : 'done',
        });
      }

      default:
        return parts;
    }
  }

  reset(): void {
    this.stop();
    this.state.set([]);
    this.failure.set(null);
    this.history = [];
  }

  stop(): void {
    this.controller?.abort();
    this.controller = null;
    this.active.set(false);
  }

  async send(modelId: string, message: string): Promise<void> {
    this.stop();
    this.failure.set(null);
    this.active.set(true);
    this.state.update((turns) => [
      ...turns,
      { role: 'user', parts: [{ kind: 'text', id: `u${turns.length}`, text: message }] },
      { role: 'assistant', parts: [] },
    ]);

    const controller = new AbortController();
    this.controller = controller;

    try {
      const response = await fetch(`/api/models/${modelId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history: this.history }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        // The refusals happen before the stream opens, so the body is JSON.
        const detail = await response
          .json()
          .then((body) => body?.detail)
          .catch(() => null);
        throw new Error(detail ?? `the server answered ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // A frame ends at a blank line; anything after the last one is partial.
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';
        for (const block of blocks) this.handle(block);
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        this.zone.run(() => this.failure.set((err as Error).message));
      }
    } finally {
      this.zone.run(() => this.active.set(false));
      this.controller = null;
    }
  }

  private handle(block: string): void {
    const lines = block.split('\n');
    const kind = lines.find((l) => l.startsWith('event: '))?.slice(7);
    const data = lines.find((l) => l.startsWith('data: '))?.slice(6);
    if (!kind || !data) return;

    const payload = JSON.parse(data) as Payload;

    // fetch resolves outside Angular's zone, so every update is wrapped to keep
    // change detection running while the stream is open.
    this.zone.run(() => {
      if (kind === 'error') {
        this.failure.set(payload.detail ?? 'the run failed');
        return;
      }
      if (kind === 'phase' && payload.phase === 'finished') {
        // R9.4 - only a run that finished contributes to the transcript. A run
        // that died mid-answer leaves the history as it was, so the next turn
        // does not build on half an exchange.
        this.history = payload.messages ?? this.history;
        return;
      }
      this.state.update((turns) => ChatStream.applyEvent(turns, kind, payload));
    });
  }
}
