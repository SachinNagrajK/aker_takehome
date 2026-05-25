// Thin API client.
//
// Dev:   VITE_API_BASE is unset → BASE='/api' → Vite proxies to local uvicorn.
// Prod:  VITE_API_BASE='https://<space>.hf.space' → calls hit HF Space directly.
//        Backend CORS_ORIGINS must include the Vercel origin in that case.
const BASE = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '')

async function json(path, opts = {}) {
  // IMPORTANT: pull `headers` out of opts BEFORE spreading the rest, otherwise
  // the spread overrides our merged header object and strips Content-Type
  // when a caller passes their own headers (e.g. the admin token).
  const { headers: optsHeaders, ...rest } = opts
  const r = await fetch(`${BASE}${path}`, {
    ...rest,
    headers: { 'Content-Type': 'application/json', ...(optsHeaders || {}) },
  })
  if (!r.ok) {
    let detail
    try { detail = (await r.json()).detail } catch { detail = await r.text() }
    throw new Error(formatErrorDetail(detail) || `HTTP ${r.status}`)
  }
  return r.json()
}

// FastAPI returns 422 with `detail` as an array of {loc, msg, type} objects.
// new Error([{...}]) stringifies to "[object Object]" — so flatten to a
// readable string before throwing.
function formatErrorDetail(detail) {
  if (!detail) return ''
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      if (typeof d === 'string') return d
      const loc = Array.isArray(d?.loc) ? d.loc.join('.') : (d?.loc || '')
      const msg = d?.msg || JSON.stringify(d)
      return loc ? `${loc}: ${msg}` : msg
    }).join('; ')
  }
  if (typeof detail === 'object') {
    return detail.msg || detail.message || JSON.stringify(detail)
  }
  return String(detail)
}

// Streaming chat — consumes the SSE endpoint at POST /chat/stream. EventSource
// can't do POST, so we use fetch+ReadableStream and parse `data: <json>\n\n`
// frames ourselves. The caller passes a single `onEvent(event)` callback
// invoked for every parsed event. Resolves when the stream ends; throws on
// network/server errors. Pass an AbortSignal to cancel mid-stream.
async function chatStream(body, { onEvent, signal } = {}) {
  const r = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) {
    let detail
    try { detail = (await r.json()).detail } catch { detail = await r.text() }
    throw new Error(detail || `HTTP ${r.status}`)
  }

  const reader = r.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      // SSE frames are separated by a blank line.
      let idx
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        // A frame is one or more lines starting with "data: ". Join the
        // data lines and parse JSON. Ignore comments/keepalives that
        // start with ":".
        const dataLines = frame
          .split('\n')
          .filter((l) => l.startsWith('data: '))
          .map((l) => l.slice(6))
        if (dataLines.length === 0) continue
        try {
          const evt = JSON.parse(dataLines.join('\n'))
          onEvent?.(evt)
        } catch (e) {
          // Tolerate the occasional malformed frame instead of killing the stream.
          // eslint-disable-next-line no-console
          console.warn('SSE parse failure', e, dataLines)
        }
      }
    }
  } finally {
    try { reader.releaseLock() } catch { /* noop */ }
  }
}

// Admin-protected eval endpoints. Caller passes the admin token; we stash it
// in the X-Admin-Token header. Token is held in sessionStorage by the UI.
const adminHeaders = (token) => (token ? { 'X-Admin-Token': token } : {})

export const api = {
  health:     () => json('/health'),
  properties: () => json('/properties'),
  llms:       () => json('/llms'),
  chat:       (body) => json('/chat', { method: 'POST', body: JSON.stringify(body) }),
  chatStream,

  evals: {
    golden:     (token) => json('/evals/golden', { headers: adminHeaders(token) }),
    listRuns:   (token, limit = 50) => json(`/evals/runs?limit=${limit}`, { headers: adminHeaders(token) }),
    getRun:     (token, id) => json(`/evals/runs/${id}`, { headers: adminHeaders(token) }),
    triggerRun: (token, body = {}) => json('/evals/runs', {
      method: 'POST', headers: adminHeaders(token), body: JSON.stringify(body),
    }),
    getSchedule: (token) => json('/evals/schedule', { headers: adminHeaders(token) }),
    putSchedule: (token, cron) => json('/evals/schedule', {
      method: 'PUT', headers: adminHeaders(token), body: JSON.stringify({ cron }),
    }),
  },
}
