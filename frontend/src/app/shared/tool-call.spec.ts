import { TestBed } from '@angular/core/testing';

import { ToolCall } from './tool-call';

/**
 * The formatting, not the markup: turning a tool call's JSON arguments into
 * something a person reads is the only logic in this component, and it has to
 * survive arguments that are not JSON at all.
 */

function toolCall(args: string, result: string | null = null, state = 'done') {
  const fixture = TestBed.createComponent(ToolCall);
  fixture.componentRef.setInput('name', 'read_card_section');
  fixture.componentRef.setInput('args', args);
  fixture.componentRef.setInput('result', result);
  fixture.componentRef.setInput('state', state);
  fixture.detectChanges();
  return fixture;
}

function text(fixture: ReturnType<typeof toolCall>): string {
  return (fixture.nativeElement as HTMLElement).textContent ?? '';
}

describe('ToolCall', () => {
  it('renders arguments as key: value, not raw JSON', () => {
    const fixture = toolCall('{"section":"Training Methodology"}');

    expect(text(fixture)).toContain('section: "Training Methodology"');
    expect(text(fixture)).not.toContain('{"section"');
  });

  it('shows arguments raw when they are not JSON', () => {
    // The case someone most needs to see is the one that would be hidden by a
    // parse failure swallowed into an empty string.
    const fixture = toolCall('not json at all');

    expect(text(fixture)).toContain('not json at all');
  });

  it('previews the first line of the result while collapsed', () => {
    const fixture = toolCall('{}', 'NVFP4 via PTQ\nsecond line nobody asked for');

    expect(text(fixture)).toContain('NVFP4 via PTQ');
    expect(text(fixture)).not.toContain('second line nobody asked for');
  });

  it('shows a failure in the collapsed line, without expanding', () => {
    // R9.9 - an error you have to go looking for is a swallowed exception.
    const fixture = toolCall('{}', 'model c/three is not in the store', 'error');

    expect(text(fixture)).toContain('not in the store');
  });

  it('carries state on the dot so it reads without expanding', () => {
    const fixture = toolCall('{}', null, 'running');
    const dot = (fixture.nativeElement as HTMLElement).querySelector('.dot');

    expect(dot?.getAttribute('data-state')).toBe('running');
  });
});
