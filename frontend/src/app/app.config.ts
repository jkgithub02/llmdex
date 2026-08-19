import {
  ApplicationConfig,
  provideBrowserGlobalErrorListeners,
  provideZoneChangeDetection,
} from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideRouter, withComponentInputBinding } from '@angular/router';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    // withComponentInputBinding is what feeds :vendor and :name into the
    // detail component's inputs. Without it the required inputs never arrive
    // and the view renders empty.
    provideRouter(routes, withComponentInputBinding()),
    // The generated client calls relative paths, so `ng serve` proxies them to
    // the backend (proxy.conf.json). Nothing here knows the backend's address.
    provideHttpClient(),
  ],
};
