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

export const api = {
  health:     () => json('/health'),
  properties: () => json('/properties'),
  llms:       () => json('/llms'),
  chat:       (body) => json('/chat', { method: 'POST', body: JSON.stringify(body) }),
  chatStream,
}
