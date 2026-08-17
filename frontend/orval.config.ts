/**
 * R7.3 - the API client is generated from the backend's OpenAPI schema, never
 * hand-written, so a change to a Pydantic model becomes a TypeScript error here.
 *
 * Refresh the schema and regenerate:
 *   curl -s http://127.0.0.1:8001/openapi.json -o openapi.json && npx orval
 */
export default {
  llmdex: {
    input: './openapi.json',
    output: {
      mode: 'split',
      target: './src/app/api/llmdex.ts',
      schemas: './src/app/api/model',
      client: 'angular',
      baseUrl: '/api',
      clean: true,
    },
  },
};
