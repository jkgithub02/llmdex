/**
 * Dev-server proxy. The generated API client calls relative paths, so nothing in
 * the app knows where the backend lives -- this is the only place that does.
 *
 * Locally the backend is on the host; under Docker Compose it is another service
 * on the compose network, so the target comes from the environment.
 */
const target = process.env['LLMDEX_API_TARGET'] || 'http://127.0.0.1:8001';

module.exports = ['/models', '/ingest', '/benchmarks', '/health'].reduce((config, path) => {
  config[path] = { target, secure: false, changeOrigin: true };
  return config;
}, {});
