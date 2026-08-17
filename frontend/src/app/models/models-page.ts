import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table';

import { LlmdexService } from '../api/llmdex.service';
import type { ModelDoc } from '../api/model/modelDoc';
import { formatBytes, formatCount } from '../field-state';

/**
 * R6.2 - the list of everything in the vault, plus the ingest box that fills it.
 *
 * The VRAM column shows a dash where the estimate is null, and says why on
 * hover: for a hybrid architecture we cannot cost correctly, refusing to print a
 * number is the product working (R2.6), not a gap in it.
 */
@Component({
  selector: 'app-models-page',
  imports: [
    FormsModule,
    RouterLink,
    MatButtonModule,
    MatCardModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatProgressBarModule,
    MatTableModule,
  ],
  template: `
    <mat-card class="ingest">
      <mat-card-header>
        <mat-card-title>Ingest a model</mat-card-title>
        <mat-card-subtitle>A Hugging Face model ID or a full URL (R1.1)</mat-card-subtitle>
      </mat-card-header>
      <mat-card-content>
        <mat-form-field appearance="outline" class="grow">
          <mat-label>Model ID or URL</mat-label>
          <input
            matInput
            [(ngModel)]="modelId"
            (keyup.enter)="ingest()"
            placeholder="Qwen/Qwen3-8B"
            [disabled]="busy()"
          />
        </mat-form-field>
        <button mat-flat-button (click)="ingest()" [disabled]="busy() || !modelId.trim()">
          Ingest
        </button>
      </mat-card-content>
      @if (busy()) {
        <mat-progress-bar mode="indeterminate" />
      }
      @if (error(); as message) {
        <p class="error"><mat-icon>error_outline</mat-icon> {{ message }}</p>
      }
    </mat-card>

    @if (models(); as rows) {
      @if (rows.length === 0) {
        <p class="empty">The vault is empty. Ingest a model to begin.</p>
      } @else {
        <table mat-table [dataSource]="rows" class="mat-elevation-z1">
          <ng-container matColumnDef="model">
            <th mat-header-cell *matHeaderCellDef>Model</th>
            <td mat-cell *matCellDef="let row">
              <a [routerLink]="['/models', row.model_id]">{{ row.model_id }}</a>
            </td>
          </ng-container>

          <ng-container matColumnDef="checkpoints">
            <th mat-header-cell *matHeaderCellDef>Checkpoints</th>
            <td mat-cell *matCellDef="let row">{{ row.checkpoints?.length ?? 0 }}</td>
          </ng-container>

          <ng-container matColumnDef="params">
            <th mat-header-cell *matHeaderCellDef>Parameters</th>
            <td mat-cell *matCellDef="let row">{{ params(row) ?? '—' }}</td>
          </ng-container>

          <ng-container matColumnDef="vram">
            <th mat-header-cell *matHeaderCellDef>VRAM estimate</th>
            <td mat-cell *matCellDef="let row" [title]="vramReason(row) ?? ''">
              {{ vram(row) ?? '—' }}
              @if (!vram(row) && vramReason(row)) {
                <mat-icon class="warn" aria-label="estimate withheld">info_outline</mat-icon>
              }
            </td>
          </ng-container>

          <ng-container matColumnDef="prose">
            <th mat-header-cell *matHeaderCellDef>Prose</th>
            <td mat-cell *matCellDef="let row">{{ proseState(row) }}</td>
          </ng-container>

          <tr mat-header-row *matHeaderRowDef="columns"></tr>
          <tr mat-row *matRowDef="let row; columns: columns"></tr>
        </table>
      }
    } @else {
      <mat-progress-bar mode="indeterminate" />
    }
  `,
  styles: `
    :host {
      display: block;
      padding: 1.5rem;
      max-width: 72rem;
      margin: 0 auto;
    }
    .ingest {
      margin-bottom: 1.5rem;
    }
    mat-card-content {
      display: flex;
      gap: 1rem;
      align-items: baseline;
    }
    .grow {
      flex: 1;
    }
    table {
      width: 100%;
    }
    .error {
      color: var(--mat-sys-error);
      display: flex;
      gap: 0.5rem;
      align-items: center;
      padding: 0 1rem 1rem;
    }
    .empty {
      opacity: 0.7;
    }
    .warn {
      font-size: 1rem;
      width: 1rem;
      height: 1rem;
      vertical-align: middle;
      opacity: 0.6;
    }
  `,
})
export class ModelsPage {
  private readonly api = inject(LlmdexService);

  protected modelId = '';
  protected readonly models = signal<ModelDoc[] | null>(null);
  protected readonly busy = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly columns = ['model', 'checkpoints', 'params', 'vram', 'prose'];

  constructor() {
    this.refresh();
  }

  protected refresh(): void {
    this.api.listModelsModelsGet().subscribe({
      next: (rows) => this.models.set(rows),
      error: (err) => this.error.set(this.message(err)),
    });
  }

  protected ingest(): void {
    const id = this.modelId.trim();
    if (!id) return;
    this.busy.set(true);
    this.error.set(null);
    this.api.ingestModelIngestPost({ model_id: id }).subscribe({
      next: () => {
        this.busy.set(false);
        this.modelId = '';
        this.refresh();
      },
      error: (err) => {
        this.busy.set(false);
        this.error.set(this.message(err));
      },
    });
  }

  protected params(doc: ModelDoc): string | null {
    return formatCount(doc.checkpoints?.[0]?.derived?.params?.total ?? null);
  }

  protected vram(doc: ModelDoc): string | null {
    return formatBytes(doc.checkpoints?.[0]?.derived?.vram?.total_bytes ?? null);
  }

  /** R2.6 - when the estimate is withheld, say what made it unreliable. */
  protected vramReason(doc: ModelDoc): string | null {
    return doc.checkpoints?.[0]?.derived?.vram?.unreliable_reason ?? null;
  }

  protected proseState(doc: ModelDoc): string {
    const checkpoint = doc.checkpoints?.[0];
    if (checkpoint?.extracted) return 'extracted';
    if (checkpoint?.manual?.quantization || checkpoint?.manual?.serving) return 'entered by hand';
    return 'not yet read';
  }

  private message(err: unknown): string {
    const detail = (err as { error?: { detail?: string }; message?: string })?.error?.detail;
    return detail ?? (err as { message?: string })?.message ?? 'request failed';
  }
}
