import { AgentStream, type AgentState } from './agent-stream';

/**
 * The reducer is tested rather than EventSource: the rules worth pinning are how
 * events become state, and those hold whatever transport delivers them.
 */

const empty: AgentState[] = [];

describe('AgentStream.applyEvent', () => {
  it('creates an agent the first time it is heard from', () => {
    const states = AgentStream.applyEvent(empty, 'phase', { agent: 'about', phase: 'started' });

    expect(states.length).toBe(1);
    expect(states[0]).toEqual(
      jasmine.objectContaining({ name: 'about', phase: 'started', reasoning: '', error: null }),
    );
  });

  it('appends reasoning rather than replacing it', () => {
    let states = AgentStream.applyEvent(empty, 'reasoning', { agent: 'about', text: 'one ' });
    states = AgentStream.applyEvent(states, 'reasoning', { agent: 'about', text: 'two' });

    expect(states[0].reasoning).toBe('one two');
  });

  it('keeps agents apart', () => {
    let states = AgentStream.applyEvent(empty, 'reasoning', { agent: 'about', text: 'A' });
    states = AgentStream.applyEvent(states, 'reasoning', { agent: 'prose', text: 'B' });

    expect(states.map((s) => s.name)).toEqual(['about', 'prose']);
    expect(states.map((s) => s.reasoning)).toEqual(['A', 'B']);
  });

  it('records an error against the agent that failed and no other', () => {
    let states = AgentStream.applyEvent(empty, 'phase', { agent: 'about', phase: 'started' });
    states = AgentStream.applyEvent(states, 'error', { agent: 'prose', detail: 'unreachable' });

    expect(states.find((s) => s.name === 'about')!.error).toBeNull();
    expect(states.find((s) => s.name === 'prose')!.error).toBe('unreachable');
  });

  it('ignores the final frame, which belongs to no agent', () => {
    const states = AgentStream.applyEvent(empty, 'phase', { agent: '', phase: 'finished' });

    expect(states).toEqual([]);
  });

  it('does not mutate the array it was given', () => {
    // The signal holds this array; mutating in place would skip change detection.
    const before = AgentStream.applyEvent(empty, 'phase', { agent: 'about', phase: 'started' });
    const after = AgentStream.applyEvent(before, 'reasoning', { agent: 'about', text: 'x' });

    expect(before[0].reasoning).toBe('');
    expect(after).not.toBe(before);
  });
});
