import { Routes } from '@angular/router';

/** R6.1 - exactly two primary tabs. Compare is a mode inside Models, not a third. */
export const routes: Routes = [
  { path: '', redirectTo: 'models', pathMatch: 'full' },
  {
    path: 'models',
    loadComponent: () => import('./models/models-page').then((m) => m.ModelsPage),
  },
  {
    // A Hugging Face ID is `vendor/name`, so it occupies two segments.
    path: 'models/:vendor/:name',
    loadComponent: () => import('./models/model-detail').then((m) => m.ModelDetail),
  },
  {
    path: 'benchmarks',
    loadComponent: () => import('./benchmarks/benchmarks-page').then((m) => m.BenchmarksPage),
  },
  { path: '**', redirectTo: 'models' },
];
