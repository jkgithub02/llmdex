/**
 * Dev-server proxy.
 *
 * The API is namespaced under /api so it cannot collide with the application's
 * own routes. It used to proxy /models directly, which meant the dev server
 * handed every navigation to `/models/...` straight to the backend and the
 * browser rendered JSON instead of the app.
 *
 * Locally the backend is on the host; under Docker Compose it is another
 * service on the compose network, so the target comes from the environment.
 */
const target = process.env['LLMDEX_API_TARGET'] || 'http://127.0.0.1:8001';

module.exports = {
  '/api': {
    target,
    secure: false,
    changeOrigin: true,
    pathRewrite: { '^/api': '' },
  },
};
