/**
 * Karma, for `npm test`.
 *
 * The Angular builder supplies almost all of this; two things it cannot know
 * are here.
 *
 * 1. Where Chrome is. karma-chrome-launcher only reads CHROME_BIN, so an
 *    already-set CHROME_BIN wins and otherwise we look in the usual places,
 *    including the browser caches Playwright and Puppeteer keep.
 * 2. --no-sandbox. Chrome's sandbox needs kernel privileges a container and
 *    most CI runners do not grant, and without the flag the browser dies on
 *    launch with a stack trace instead of a test result. Safe here: the only
 *    page it ever loads is our own test bundle from localhost.
 */

const { existsSync, readdirSync } = require('node:fs');
const { homedir } = require('node:os');
const { join } = require('node:path');

function findChrome() {
  if (process.env['CHROME_BIN']) return process.env['CHROME_BIN'];

  const cache = join(homedir(), '.cache', 'ms-playwright');
  const playwright = existsSync(cache)
    ? readdirSync(cache)
        .filter((entry) => entry.startsWith('chromium-'))
        .map((entry) => join(cache, entry, 'chrome-linux64', 'chrome'))
    : [];

  return [
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/google-chrome',
    '/opt/google/chrome/chrome',
    ...playwright,
  ].find(existsSync);
}

module.exports = (config) => {
  const chrome = findChrome();
  if (chrome) process.env['CHROME_BIN'] = chrome;

  config.set({
    frameworks: ['jasmine'],
    plugins: [require('karma-jasmine'), require('karma-chrome-launcher')],
    customLaunchers: {
      ChromeHeadlessNoSandbox: {
        base: 'ChromeHeadless',
        flags: ['--no-sandbox', '--disable-gpu'],
      },
    },
    browsers: ['ChromeHeadlessNoSandbox'],
  });
};
