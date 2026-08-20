import { ChatStream, type Turn } from './chat-stream';

/**
 * The reducer is tested rather than fetch: the rules worth pinning are how
 * frames become turns, and those hold whatever transport delivers them.
 *
 * The sequence below is the one measured against the live endpoint
 * (research.md 6e) -- thinking, an empty text part, tool calls, results, then
 * the answer with its index reset.
 */

const open = (): Turn[] => [{ role: 'assistant', parts: [] }];

describe('ChatStream.applyEvent', () => {
  it('opens a thinking part and appends its deltas', () => {
    let turns = ChatStream.applyEvent(open(), 'part_start', { part_id: '0:0', part: 'thinking' });
    turns = ChatStream.applyEvent(turns, 'reasoning', { part_id: '0:0', text: 'hm ' });
    turns = ChatStream.applyEvent(turns, 'reasoning', { part_id: '0:0', text: 'ok' });

    const part = turns[0].parts[0];
    expect(part.kind).toBe('thinking');
    expect(part.kind === 'thinking' && part.text).toBe('hm ok');
    expect(part.kind === 'thinking' && part.done).toBe(false);
  });

  it('marks a thinking part done on part_end so it can collapse', () => {
    let turns = ChatStream.applyEvent(open(), 'part_start', { part_id: '0:0', part: 'thinking' });
    turns = ChatStream.applyEvent(turns, 'reasoning', { part_id: '0:0', text: 'thought' });
    turns = ChatStream.applyEvent(turns, 'part_end', { part_id: '0:0' });

    const part = turns[0].parts[0];
    expect(part.kind === 'thinking' && part.done).toBe(true);
  });

  it('keeps parts in the order they arrived', () => {
    let turns = ChatStream.applyEvent(open(), 'part_start', { part_id: '0:0', part: 'thinking' });
    turns = ChatStream.applyEvent(turns, 'tool_call', {
      tool_call_id: 'c1',
      tool: 'grep_card',
      args: '{"query":"H100"}',
    });
    turns = ChatStream.applyEvent(turns, 'part_start', { part_id: '1:0', part: 'text' });
    turns = ChatStream.applyEvent(turns, 'content', { part_id: '1:0', text: 'yes' });

    expect(turns[0].parts.map((p) => p.kind)).toEqual(['thinking', 'tool', 'text']);
  });

  it('keeps parts apart when the index resets between rounds', () => {
    let turns = ChatStream.applyEvent(open(), 'part_start', { part_id: '0:0', part: 'thinking' });
    turns = ChatStream.applyEvent(turns, 'reasoning', { part_id: '0:0', text: 'thought' });
    turns = ChatStream.applyEvent(turns, 'part_start', { part_id: '1:0', part: 'text' });
    turns = ChatStream.applyEvent(turns, 'content', { part_id: '1:0', text: 'answer' });

    expect(turns[0].parts.length).toBe(2);
    const [thinking, text] = turns[0].parts;
    expect(thinking.kind === 'thinking' && thinking.text).toBe('thought');
    expect(text.kind === 'text' && text.text).toBe('answer');
  });

  it('drops a text part that never received a delta', () => {
    // Measured: one appears before the tool calls on every run.
    let turns = ChatStream.applyEvent(open(), 'part_start', { part_id: '0:1', part: 'text' });
    turns = ChatStream.applyEvent(turns, 'part_end', { part_id: '0:1' });

    expect(turns[0].parts.length).toBe(0);
  });

  it('keeps a thinking part that never received a delta', () => {
    // Only empty *text* parts are noise. An empty thinking part still says the
    // model thought, and dropping it would make the turn misreport itself.
    let turns = ChatStream.applyEvent(open(), 'part_start', { part_id: '0:0', part: 'thinking' });
    turns = ChatStream.applyEvent(turns, 'part_end', { part_id: '0:0' });

    expect(turns[0].parts.length).toBe(1);
  });

  it('opens a tool part running and closes it with its result', () => {
    let turns = ChatStream.applyEvent(open(), 'tool_call', {
      tool_call_id: 'c1',
      tool: 'read_card_section',
      args: '{"section":"Training Methodology"}',
    });

    let part = turns[0].parts[0];
    expect(part.kind === 'tool' && part.state).toBe('running');

    turns = ChatStream.applyEvent(turns, 'tool_result', {
      tool_call_id: 'c1',
      result: 'NVFP4 via PTQ',
      ok: true,
    });

    part = turns[0].parts[0];
    expect(part.kind === 'tool' && part.state).toBe('done');
    expect(part.kind === 'tool' && part.result).toBe('NVFP4 via PTQ');
  });

  it('pairs a result to its call by id, not by position', () => {
    let turns = ChatStream.applyEvent(open(), 'tool_call', {
      tool_call_id: 'c1',
      tool: 'grep_card',
      args: '{}',
    });
    turns = ChatStream.applyEvent(turns, 'tool_call', {
      tool_call_id: 'c2',
      tool: 'list_models',
      args: '{}',
    });
    turns = ChatStream.applyEvent(turns, 'tool_result', {
      tool_call_id: 'c2',
      result: 'two models',
      ok: true,
    });

    const [first, second] = turns[0].parts;
    expect(first.kind === 'tool' && first.state).toBe('running');
    expect(second.kind === 'tool' && second.result).toBe('two models');
  });

  it('marks a failed tool call so it reads without expanding', () => {
    let turns = ChatStream.applyEvent(open(), 'tool_call', {
      tool_call_id: 'c1',
      tool: 'read_model',
      args: '{}',
    });
    turns = ChatStream.applyEvent(turns, 'tool_result', {
      tool_call_id: 'c1',
      result: 'boom',
      ok: false,
    });

    const part = turns[0].parts[0];
    expect(part.kind === 'tool' && part.state).toBe('error');
  });

  it('never mutates the array it was given', () => {
    // The signal holds that array; mutating in place would skip change detection.
    const before = open();
    const after = ChatStream.applyEvent(before, 'part_start', { part_id: '0:0', part: 'text' });

    expect(before[0].parts.length).toBe(0);
    expect(after).not.toBe(before);
  });

  it('ignores a frame when no assistant turn is open', () => {
    const turns = ChatStream.applyEvent([], 'content', { part_id: '0:0', text: 'stray' });

    expect(turns).toEqual([]);
  });

  it('replays the full measured sequence into three rendered parts', () => {
    // research.md 6e end to end: think, empty text, two tools, then the answer.
    let turns = open();
    const send = (kind: string, payload: Record<string, unknown>) => {
      turns = ChatStream.applyEvent(turns, kind, payload);
    };

    send('part_start', { part_id: '0:0', part: 'thinking' });
    send('reasoning', { part_id: '0:0', text: 'which sections?' });
    send('part_end', { part_id: '0:0' });
    send('part_start', { part_id: '0:1', part: 'text' });
    send('part_end', { part_id: '0:1' });
    send('tool_call', { tool_call_id: 'c1', tool: 'read_card_section', args: '{}' });
    send('tool_call', { tool_call_id: 'c2', tool: 'read_card_section', args: '{}' });
    send('tool_result', { tool_call_id: 'c1', result: 'NVFP4', ok: true });
    send('tool_result', { tool_call_id: 'c2', result: 'vLLM 0.27.1', ok: true });
    send('part_start', { part_id: '1:0', part: 'text' });
    send('content', { part_id: '1:0', text: 'It uses NVFP4 and needs vLLM 0.27.1.' });
    send('part_end', { part_id: '1:0' });

    expect(turns[0].parts.map((p) => p.kind)).toEqual(['thinking', 'tool', 'tool', 'text']);
    const answer = turns[0].parts[3];
    expect(answer.kind === 'text' && answer.text).toContain('NVFP4');
  });
});
