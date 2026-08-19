import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { ModelDetail } from './model-detail';

/**
 * rerun() starts an effect() from a click handler, not a constructor -- the
 * one case that trips NG0203 (effect() outside an injection context) without
 * the build or any other spec noticing. This pins that it does not throw.
 */
class FakeEventSource {
  onerror: (() => void) | null = null;
  addEventListener(): void {}
  close(): void {}
}

describe('ModelDetail.rerun', () => {
  let fixture: ComponentFixture<ModelDetail>;
  let originalEventSource: typeof EventSource;

  beforeEach(async () => {
    originalEventSource = window.EventSource;
    (window as unknown as { EventSource: unknown }).EventSource = FakeEventSource;

    await TestBed.configureTestingModule({
      imports: [ModelDetail],
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ModelDetail);
    fixture.componentRef.setInput('vendor', 'a');
    fixture.componentRef.setInput('name', 'one');
  });

  afterEach(() => {
    window.EventSource = originalEventSource;
  });

  it('does not throw NG0203 when run from a click handler', () => {
    expect(() => (fixture.componentInstance as any).rerun('about')).not.toThrow();
  });
});
