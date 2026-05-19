// Thin API client. The Vite dev server proxies /api/* to the FastAPI backend
// at http://127.0.0.1:8000. In production you'd set this to the real URL.
const BASE = '/api'

async function json(path, opts = {}) {
  const r = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  })
  if (!r.ok) {
    let detail
    try { detail = (await r.json()).detail } catch { detail = await r.text() }
    throw new Error(detail || `HTTP ${r.status}`)
  }
  return r.json()
}

export const api = {
  health:     () => json('/health'),
  properties: () => json('/properties'),
  llms:       () => json('/llms'),
  chat:       (body) => json('/chat', { method: 'POST', body: JSON.stringify(body) }),
}
