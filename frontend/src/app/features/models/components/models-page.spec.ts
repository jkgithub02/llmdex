import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { ModelsPage } from './models-page';

/**
 * The rule under test is a spending rule: a re-ingest refreshes config-derived
 * facts and must not pay for LLM calls doing it, nor overwrite a summary
 * somebody regenerated on purpose (R6.5). The backend keeps the same rule in
 * summarise_after_first_ingest; this is the half that lives in the wiring.
 */
describe('ModelsPage first-ingest rule', () => {
  function page(): any {
    TestBed.configureTestingModule({
      imports: [ModelsPage],
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    });
    const fixture = TestBed.createComponent(ModelsPage);
    return fixture.componentInstance as any;
  }

  it('treats a model already in the vault as known', () => {
    const component = page();
    component.models.set([{ model_id: 'Qwen/Qwen3-8B' }]);

    expect(component.knows('Qwen/Qwen3-8B')).toBe(true);
  });

  it('treats a model it has never seen as new', () => {
    const component = page();
    component.models.set([{ model_id: 'Qwen/Qwen3-8B' }]);

    expect(component.knows('nvidia/Nemotron-H-8B-Base-8K')).toBe(false);
  });

  it('sees through a pasted Hub URL, which the ingest box accepts', () => {
    const component = page();
    component.models.set([{ model_id: 'Qwen/Qwen3-8B' }]);

    expect(component.knows('https://huggingface.co/Qwen/Qwen3-8B')).toBe(true);
    expect(component.knows('  qwen/qwen3-8b/  ')).toBe(true);
  });

  it('is false against an empty vault', () => {
    const component = page();
    component.models.set([]);

    expect(component.knows('Qwen/Qwen3-8B')).toBe(false);
  });
});
