// Where the API lives, seen from the browser. In development Vite forwards /api/... to http://127.0.0.1:4004 (see vite.config.ts).
export const env = {
  apiUrl: '/api',
  apiStartCommand: 'python app.py',
}
