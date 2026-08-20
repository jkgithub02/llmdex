import { TestBed } from '@angular/core/testing';

import { Markdown } from './markdown';

/**
 * Two things matter here: that a comparison table renders as a table (the
 * reason this component exists), and that Angular's sanitizer is still doing
 * its job on text a language model wrote.
 */

function render(text: string): HTMLElement {
  const fixture = TestBed.createComponent(Markdown);
  fixture.componentRef.setInput('text', text);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('Markdown', () => {
  it('renders a comparison table as a table', () => {
    const el = render(
      ['| Benchmark | Pro | Flash |', '| --- | --- | --- |', '| DeepSWE | 62.7 | 54.4 |'].join(
        '\n',
      ),
    );

    expect(el.querySelector('table')).toBeTruthy();
    expect(el.querySelectorAll('th').length).toBe(3);
    expect(el.querySelector('td')?.textContent).toBe('DeepSWE');
  });

  it('wraps a table so it scrolls inside the panel', () => {
    // Without this a wide table pushes the whole layout sideways.
    const el = render('| a | b |\n| --- | --- |\n| 1 | 2 |');

    expect(el.querySelector('.table-wrap table')).toBeTruthy();
  });

  it('renders emphasis, lists and code', () => {
    const el = render('**bold** and `code`\n\n- one\n- two');

    expect(el.querySelector('strong')?.textContent).toBe('bold');
    expect(el.querySelector('code')?.textContent).toBe('code');
    expect(el.querySelectorAll('li').length).toBe(2);
  });

  it('strips a script the model was talked into emitting', () => {
    // Angular sanitizes [innerHTML]; this asserts we have not bypassed it.
    const el = render('before\n\n<script>window.pwned = 1</script>\n\nafter');

    expect(el.querySelector('script')).toBeNull();
    expect((window as unknown as Record<string, unknown>)['pwned']).toBeUndefined();
  });

  it('does not pass raw html through', () => {
    const el = render('<img src=x onerror="window.pwned = 1">');

    expect(el.querySelector('img[onerror]')).toBeNull();
  });

  it('renders plain prose without decoration', () => {
    const el = render('It uses NVFP4 via PTQ.');

    expect(el.textContent).toContain('It uses NVFP4 via PTQ.');
  });
});
