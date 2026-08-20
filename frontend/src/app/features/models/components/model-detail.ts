import { Component, Injector, computed, effect, inject, input, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { AgentStream } from '../../../shared/agent-stream';
import { ChatStream } from '../../../shared/chat-stream';
import { LlmdexService } from '../../../api/llmdex.service';
import type { Checkpoint } from '../../../api/model/checkpoint';
import type { ModelDoc } from '../../../api/model/modelDoc';
import { errorMessage, formatCount } from '../../../shared/format';
import { architecturePills, architectureTone } from '../../../shared/architecture';
import { AboutTab } from './about-tab';
import { BenchmarksTab } from './benchmarks-tab';
import { DetailsTab } from './details-tab';
import { LayerStrip } from '../../../shared/layer-strip';
import { ChatPanel } from './chat-panel';
import { SpecTab } from './spec-tab';

type Tab = 'about' | 'spec' | 'prose' | 'benchmarks'; // 'prose' is the agent's name on the wire

// The shell (R6.3 / R6.4): a Pokedex-style hero over a light sheet of tabbed
// detail. Each tab owns its own template, styles and helpers; this owns only
// what every tab shares -- loading, the agent-driven mutations, the error banner.
@Component({
  selector: 'app-model-detail',
  imports: [RouterLink, LayerStrip, AboutTab, BenchmarksTab, SpecTab, DetailsTab, ChatPanel],
  host: { class: 'page' },
  template: `
    <!-- Outside the @if below: a failure after load must still show up. -->
    @if (error(); as message) {
      <p class="error" role="alert">
        <strong>{{ errorLabel() }}</strong>
        {{ message }}
      </p>
    }

    @if (doc(); as model) {
      <div class="split" [class.collapsed]="!chatOpen()">
        <article class="dex" [attr.data-tone]="tone()">
          <header class="hero">
            <div class="hero-top">
              <a routerLink="/models" class="back" aria-label="back to models">←</a>
              <span class="vendor">{{ vendor() }}</span>
              <span class="params mono">{{ params() ?? 'size unavailable' }}</span>
            </div>

            <h1 class="mono">{{ name() }}</h1>

            <div class="pills">
              @for (pill of pills(); track pill) {
                <span class="pill">{{ pill }}</span>
              }
            </div>

            <app-layer-strip [layers]="primary()?.derived?.layers" />
          </header>

          <div class="sheet">
            <nav class="tabs" role="tablist">
              @for (t of tabs; track t.id) {
                <button
                  role="tab"
                  [class.on]="tab() === t.id"
                  [attr.aria-selected]="tab() === t.id"
                  (click)="tab.set(t.id)"
                >
                  {{ t.label }}
                </button>
              }
            </nav>

            @switch (tab()) {
              @case ('about') {
                <app-about-tab
                  [model]="model"
                  [modelId]="modelId()"
                  [busy]="busy()"
                  [summarising]="summarising()"
                  (rerun)="rerun($event)"
                  (summarise)="summarise($event)"
                />
              }
              @case ('spec') {
                <app-spec-tab [model]="model" />
              }
              @case ('prose') {
                <app-details-tab
                  [model]="model"
                  [modelId]="modelId()"
                  [busy]="busy()"
                  [extracting]="extracting()"
                  (rerun)="rerun($event)"
                />
              }
              @case ('benchmarks') {
                <app-benchmarks-tab
                  [model]="model"
                  [modelId]="modelId()"
                  [busy]="busy()"
                  (rerun)="rerun($event)"
                />
              }
            }
          </div>
        </article>

        <aside class="side">
          <button class="handle" (click)="chatOpen.set(!chatOpen())">
            {{ chatOpen() ? 'hide ›' : '‹ Ask' }}
          </button>
          @if (chatOpen()) {
            <app-chat-panel [modelId]="modelId()" />
          }
        </aside>
      </div>
    } @else if (!error()) {
      <div class="bar"><span></span></div>
    }
  `,
  styles: `
    :host(.page) {
      max-width: 96rem;
      padding-top: var(--space-4);
    }

    /* The sheet keeps its 60rem measure; the panel takes what is left. */
    .split {
      display: grid;
      grid-template-columns: minmax(0, 60rem) minmax(20rem, 26rem);
      gap: var(--space-4);
      align-items: start;
    }
    .split.collapsed {
      grid-template-columns: minmax(0, 60rem) auto;
    }
    .side {
      display: flex;
      flex-direction: column;
      gap: var(--space-2);
      position: sticky;
      top: var(--space-4);
      max-height: calc(100vh - var(--space-6));
      min-height: 0;
    }
    .side app-chat-panel {
      flex: 1;
      min-height: 0;
    }
    .handle {
      align-self: flex-start;
      background: none;
      border: none;
      color: var(--sheet-faint);
      font-size: 0.78rem;
      padding: 0;
      white-space: nowrap;
    }
    .handle:hover {
      color: var(--sheet-fg);
    }

    /* One column below this: the sheet needs the width more than the panel. */
    @media (max-width: 1100px) {
      :host(.page) {
        max-width: 60rem;
      }
      .split,
      .split.collapsed {
        grid-template-columns: minmax(0, 1fr);
      }
      .side {
        position: static;
        max-height: none;
      }
      .side app-chat-panel {
        height: 32rem;
      }
    }

    /* Set once on the article; everything coloured reads it from here. */
    .dex[data-tone='transformer'] {
      --tone: var(--type-transformer);
    }
    .dex[data-tone='hybrid'] {
      --tone: var(--type-hybrid);
    }
    .dex[data-tone='recurrent'] {
      --tone: var(--type-recurrent);
    }
    .dex[data-tone='unknown'] {
      --tone: var(--type-unknown);
    }
    .hero {
      background: var(--tone);
      background-image: radial-gradient(
        120% 80% at 80% 0%,
        rgb(255 255 255 / 0.18),
        transparent 60%
      );
      border-radius: var(--radius) var(--radius) 0 0;
      padding: var(--space-4) var(--space-6) var(--space-6);
      color: #fff;
    }
    .hero-top {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      font-size: 0.8rem;
    }
    .back {
      color: #fff;
      font-size: 1.1rem;
      line-height: 1;
      opacity: 0.85;
    }
    .back:hover {
      opacity: 1;
    }
    .hero .vendor {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      opacity: 0.85;
      color: #fff;
    }
    .params {
      margin-left: auto;
      font-weight: 500;
      opacity: 0.9;
    }
    h1 {
      margin: var(--space-2) 0 var(--space-3);
      font-size: 1.9rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      overflow-wrap: anywhere;
    }
    .pills {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
      margin-bottom: var(--space-6);
    }
    .pill {
      background: rgb(255 255 255 / 0.22);
      border: 1px solid rgb(255 255 255 / 0.25);
      border-radius: 999px;
      padding: 0.15rem 0.7rem;
      font-size: 0.75rem;
      font-weight: 500;
    }
    .sheet {
      background: var(--sheet);
      color: var(--sheet-fg);
      border-radius: var(--radius);
      margin-top: calc(var(--space-4) * -1);
      padding: var(--space-2) var(--space-6) var(--space-6);
      position: relative;
    }
    .tabs {
      display: flex;
      gap: var(--space-4);
      border-bottom: 1px solid var(--sheet-border);
      margin-bottom: var(--space-5, 1.25rem);
      overflow-x: auto;
    }
    .tabs button {
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      color: var(--sheet-faint);
      font-size: 0.85rem;
      font-weight: 600;
      padding: var(--space-4) 0 var(--space-3);
      white-space: nowrap;
    }
    .tabs button:hover {
      color: var(--sheet-fg);
    }
    .tabs button.on {
      color: var(--tone);
      border-bottom-color: var(--tone);
    }
  `,
})
export class ModelDetail {
  private readonly api = inject(LlmdexService);
  private readonly stream = inject(AgentStream);
  private readonly chat = inject(ChatStream);
  private readonly injector = inject(Injector);

  /** A Hugging Face ID is `vendor/name`, so it arrives as two route segments. */
  readonly vendor = input.required<string>();
  readonly name = input.required<string>();

  protected readonly doc = signal<ModelDoc | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly errorLabel = signal('Extraction failed.');
  protected readonly extracting = signal(false);
  protected readonly summarising = signal(false);
  protected readonly tab = signal<Tab>('about');
  protected readonly chatOpen = signal(true);
  protected readonly modelId = computed(() => `${this.vendor()}/${this.name()}`);

  protected readonly tabs: { id: Tab; label: string }[] = [
    { id: 'about', label: 'About' },
    { id: 'spec', label: 'Spec' },
    { id: 'prose', label: 'Details' },
    { id: 'benchmarks', label: 'Benchmarks' },
  ];

  constructor() {
    // ChatStream is root-provided, so a new model must start a new
    // conversation rather than inherit the last one's history -- which would
    // read as the model hallucinating about a card it never saw.
    effect(() => {
      this.modelId();
      this.chat.reset();
    });
    queueMicrotask(() => this.load());
  }

  // The checkpoint the hero describes: model properties, not per-checkpoint ones.
  protected readonly primary = computed<Checkpoint | undefined>(() => this.doc()?.checkpoints?.[0]);

  protected readonly pills = computed(() =>
    architecturePills(this.primary()?.derived?.architecture_class),
  );
  protected readonly tone = computed(() =>
    architectureTone(this.primary()?.derived?.architecture_class),
  );
  protected readonly params = computed(() => formatCount(this.primary()?.derived?.params?.total));
  protected readonly busy = computed(() => this.summarising() || this.extracting());

  protected load(): void {
    this.api.getModelModelsModelIdGet(this.modelId()).subscribe({
      next: (doc) => this.doc.set(doc),
      error: (err) => {
        this.errorLabel.set('Could not load this model.');
        this.error.set(errorMessage(err));
      },
    });
  }

  // R6.x - regenerating replaces what is on screen; `generated_on` says which run wrote it.
  protected summarise(modelId: string): void {
    this.summarising.set(true);
    this.error.set(null);
    this.errorLabel.set('Summary failed.');
    this.api.summariseModelModelsModelIdSummarizePost(modelId).subscribe({
      next: (doc) => {
        this.summarising.set(false);
        this.doc.set(doc);
      },
      error: (err) => {
        this.summarising.set(false);
        this.error.set(errorMessage(err));
      },
    });
  }

  // Re-run a tab's agent, then reload. Explicit injector: run from a click handler --
  // effect() outside an injection context throws NG0203 at runtime.
  protected rerun(agent: string): void {
    this.stream.start(this.modelId(), [agent]);
    const finished = effect(
      () => {
        if (!this.stream.running()) {
          this.load();
          finished.destroy();
        }
      },
      { injector: this.injector },
    );
  }
}
