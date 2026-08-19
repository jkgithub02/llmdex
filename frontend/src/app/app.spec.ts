import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { App } from './app';

describe('App', () => {
  let fixture: ComponentFixture<App>;
  let element: HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    element = fixture.nativeElement as HTMLElement;
  });

  it('renders topbar navigation links', () => {
    const links = element.querySelectorAll('nav a');
    expect(links.length).toBe(2);
    expect(links[0].textContent?.trim()).toBe('Models');
    expect(links[1].textContent?.trim()).toBe('Benchmarks');
  });

  it('does not render the old tagline', () => {
    expect(element.querySelector('.tagline')).toBeNull();
    expect(element.textContent).not.toContain('derived · extracted · absent · unmeasured');
  });

  it('renders footer with developer, last updated, and version details', () => {
    const footer = element.querySelector('footer.app-footer');
    expect(footer).not.toBeNull();
    const footerLeft = footer?.querySelector('.footer-left');
    const footerRight = footer?.querySelector('.footer-right');
    expect(footerLeft?.textContent).toContain('jason.kong');
    expect(footerRight?.textContent).toContain('August 2026');
    expect(footerRight?.querySelector('.version-tag')?.textContent?.trim()).toBe('V0.1.0');
  });
});
