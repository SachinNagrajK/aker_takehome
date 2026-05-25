import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'

const TOKEN_KEY = 'aker.admin_token'

function fmtTime(iso) {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

function fmtScore(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v !== 'number') return String(v)
  return v.toFixed(2)
}

function scoreColor(v) {
  if (typeof v !== 'number') return 'inherit'
  if (v >= 0.8) return '#1b8f4a'
  if (v >= 0.5) return '#b88600'
  return '#c53030'
}

export default function Monitoring() {
  // `token` is the committed value used for API calls. `tokenDraft` is what
  // the input is bound to — promoted to `token` only on form submit so we
  // don't hit /evals/* on every keystroke (which 401s on a partial token
  // and used to spam errors).
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY) || '')
  const [tokenDraft, setTokenDraft] = useState(() => sessionStorage.getItem(TOKEN_KEY) || '')
  const [golden, setGolden] = useState([])
  const [selectedIds, setSelectedIds] = useState([])
  const [runs, setRuns] = useState([])
  const [openRunId, setOpenRunId] = useState(null)
  const [openRunDetail, setOpenRunDetail] = useState(null)
  const [schedule, setSchedule] = useState(null)
  const [cronDraft, setCronDraft] = useState('')
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)

  const authed = !!token

  const refresh = useCallback(async () => {
    if (!token) return
    setErr(null)
    try {
      const [g, r, s] = await Promise.all([
        api.evals.golden(token),
        api.evals.listRuns(token),
        api.evals.getSchedule(token),
      ])
      setGolden(g)
      setRuns(r)
      setSchedule(s)
      setCronDraft(s.cron || '')
    } catch (e) {
      // 401 here means the saved token is stale — drop it back to the
      // login view instead of showing a noisy error.
      if (/401|Invalid admin token/i.test(e.message || '')) {
        sessionStorage.removeItem(TOKEN_KEY)
        setToken('')
        setErr('Saved token rejected. Re-enter it.')
      } else {
        setErr(e.message)
      }
    }
  }, [token])

  useEffect(() => { refresh() }, [refresh])

  // Light polling while any run is still going.
  useEffect(() => {
    if (!token) return
    const hasRunning = runs.some((r) => r.status === 'running')
    if (!hasRunning) return
    const t = setInterval(refresh, 3000)
    return () => clearInterval(t)
  }, [runs, token, refresh])

  async function saveToken(e) {
    e.preventDefault()
    const t = (tokenDraft || '').trim()
    if (!t) return
    sessionStorage.setItem(TOKEN_KEY, t)
    setToken(t)            // triggers refresh() via the useEffect on `token`
  }

  function clearToken() {
    sessionStorage.removeItem(TOKEN_KEY)
    setToken('')
    setTokenDraft('')
    setGolden([]); setRuns([]); setSchedule(null)
    setErr(null)
  }

  async function triggerRun() {
    setBusy(true); setErr(null)
    try {
      const body = selectedIds.length > 0 ? { ids: selectedIds } : {}
      const { run_id } = await api.evals.triggerRun(token, body)
      setOpenRunId(run_id)
      await refresh()
    } catch (e) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function openRun(id) {
    setOpenRunId(id)
    setOpenRunDetail(null)
    try {
      const detail = await api.evals.getRun(token, id)
      setOpenRunDetail(detail)
    } catch (e) {
      setErr(e.message)
    }
  }

  async function saveCron(e) {
    e.preventDefault()
    setErr(null)
    try {
      const s = await api.evals.putSchedule(token, cronDraft.trim())
      setSchedule(s)
    } catch (e) {
      setErr(e.message)
    }
  }

  const toggleId = (id) => {
    setSelectedIds((cur) => cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id])
  }

  const allSelected = selectedIds.length === golden.length && golden.length > 0

  if (!authed) {
    return (
      <div style={{ padding: 24, maxWidth: 480 }}>
        <h2>Monitoring</h2>
        <p style={{ color: 'var(--muted)' }}>Enter the backend admin token to view eval runs and trigger evaluations.</p>
        <form onSubmit={saveToken} style={{ display: 'flex', gap: 8 }}>
          <input
            type="password"
            placeholder="X-Admin-Token"
            value={tokenDraft}
            onChange={(e) => setTokenDraft(e.target.value)}
            autoComplete="off"
            style={{ flex: 1, padding: 8 }}
          />
          <button type="submit" disabled={!tokenDraft.trim()}>Save</button>
        </form>
        {err && <div className="error" style={{ marginTop: 12 }}>{err}</div>}
      </div>
    )
  }

  return (
    <div style={{ padding: 16, overflow: 'auto', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Monitoring & Evaluation</h2>
        <button onClick={refresh} disabled={busy}>Refresh</button>
        <button onClick={clearToken} style={{ marginLeft: 'auto' }}>Forget token</button>
      </div>

      {err && <div className="error" style={{ marginBottom: 12 }}>{err}</div>}

      {/* SCHEDULE */}
      <section style={{ marginBottom: 24, padding: 12, border: '1px solid var(--border, #ddd)', borderRadius: 8 }}>
        <h3 style={{ marginTop: 0 }}>Schedule</h3>
        {schedule ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 6, fontSize: 14 }}>
              <div style={{ color: 'var(--muted)' }}>Enabled</div><div>{String(schedule.enabled)}</div>
              <div style={{ color: 'var(--muted)' }}>Running</div><div>{String(schedule.running)}</div>
              <div style={{ color: 'var(--muted)' }}>Next run</div><div>{fmtTime(schedule.next_run_at)}</div>
            </div>
            <form onSubmit={saveCron} style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <input
                value={cronDraft}
                onChange={(e) => setCronDraft(e.target.value)}
                placeholder="cron e.g. 0 */6 * * *"
                style={{ flex: 1, padding: 6, fontFamily: 'monospace' }}
                disabled={!schedule.running}
              />
              <button type="submit" disabled={!schedule.running}>Save cron</button>
            </form>
            {!schedule.running && (
              <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>
                Scheduler not running. Set <code>EVAL_SCHEDULE_ENABLED=true</code> in the backend env to enable.
              </div>
            )}
          </>
        ) : <div>Loading…</div>}
      </section>

      {/* TRIGGER */}
      <section style={{ marginBottom: 24, padding: 12, border: '1px solid var(--border, #ddd)', borderRadius: 8 }}>
        <h3 style={{ marginTop: 0 }}>Trigger run</h3>
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 13 }}>
            <input
              type="checkbox"
              checked={allSelected}
              onChange={() => setSelectedIds(allSelected ? [] : golden.map((g) => g.id))}
            />{' '}
            Select all ({golden.length})
          </label>
        </div>
        <div style={{ maxHeight: 220, overflow: 'auto', border: '1px solid #eee', padding: 8, marginBottom: 12 }}>
          {golden.map((g) => (
            <div key={g.id} style={{ display: 'flex', gap: 8, padding: 2, fontSize: 13 }}>
              <input
                type="checkbox"
                checked={selectedIds.includes(g.id)}
                onChange={() => toggleId(g.id)}
              />
              <code style={{ minWidth: 180 }}>{g.id}</code>
              <span style={{ color: 'var(--muted)' }}>{g.property_code}</span>
              <span>{g.question}</span>
            </div>
          ))}
        </div>
        <button onClick={triggerRun} disabled={busy}>
          {busy ? 'Triggering…' : selectedIds.length > 0 ? `Run ${selectedIds.length} selected` : 'Run all evals'}
        </button>
      </section>

      {/* HISTORY */}
      <section style={{ padding: 12, border: '1px solid var(--border, #ddd)', borderRadius: 8 }}>
        <h3 style={{ marginTop: 0 }}>Run history</h3>
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>
              <th>Started</th><th>Trigger</th><th>Status</th><th>Cases</th>
              <th>Groundedness</th><th>Hallucination</th><th>Ans-relev</th><th>Ctx-relev</th><th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const s = r.summary || {}
              return (
                <tr key={r.id} style={{ borderBottom: '1px solid #eee', cursor: 'pointer' }}
                    onClick={() => openRun(r.id)}>
                  <td>{fmtTime(r.started_at)}</td>
                  <td>{r.trigger}</td>
                  <td>{r.status}</td>
                  <td>{s.count ?? '—'}</td>
                  <td style={{ color: scoreColor(s.mean_groundedness) }}>{fmtScore(s.mean_groundedness)}</td>
                  <td style={{ color: scoreColor(s.mean_hallucination) }}>{fmtScore(s.mean_hallucination)}</td>
                  <td style={{ color: scoreColor(s.mean_answer_relevance) }}>{fmtScore(s.mean_answer_relevance)}</td>
                  <td style={{ color: scoreColor(s.mean_context_relevance) }}>{fmtScore(s.mean_context_relevance)}</td>
                  <td><button onClick={(e) => { e.stopPropagation(); openRun(r.id) }}>Open</button></td>
                </tr>
              )
            })}
            {runs.length === 0 && (
              <tr><td colSpan={9} style={{ padding: 12, color: 'var(--muted)' }}>No runs yet.</td></tr>
            )}
          </tbody>
        </table>

        {openRunId && openRunDetail && openRunDetail.id === openRunId && (
          <RunDetail run={openRunDetail} onClose={() => { setOpenRunId(null); setOpenRunDetail(null) }} />
        )}
      </section>
    </div>
  )
}

function RunDetail({ run, onClose }) {
  return (
    <div style={{ marginTop: 16, padding: 12, background: '#fafafa', borderRadius: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ margin: 0 }}>Run {run.id.slice(0, 8)} · {run.trigger}</h4>
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--muted)', margin: '4px 0' }}>
        {fmtTime(run.started_at)} → {fmtTime(run.finished_at)} · status: {run.status}
      </div>
      <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse', marginTop: 8 }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>
            <th>Case</th><th>Property</th><th>OK</th>
            <th>G</th><th>H</th><th>AR</th><th>CR</th>
            <th>Trace</th>
          </tr>
        </thead>
        <tbody>
          {(run.cases || []).map((c) => {
            const s = c.scores || {}
            return (
              <tr key={c.golden_id} style={{ borderBottom: '1px solid #eee', verticalAlign: 'top' }}>
                <td>
                  <div><code>{c.golden_id}</code></div>
                  <div style={{ color: 'var(--muted)' }}>{c.question}</div>
                  {c.answer && (
                    <details style={{ marginTop: 4 }}>
                      <summary style={{ cursor: 'pointer', color: 'var(--muted)' }}>answer</summary>
                      <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}>{c.answer}</pre>
                    </details>
                  )}
                  {c.error && <div style={{ color: '#c53030' }}>{c.error}</div>}
                </td>
                <td>{c.property_code}</td>
                <td>{c.ok ? '✓' : '✗'}</td>
                <td style={{ color: scoreColor(s.groundedness) }}>{fmtScore(s.groundedness)}</td>
                <td style={{ color: scoreColor(s.hallucination) }}>{fmtScore(s.hallucination)}</td>
                <td style={{ color: scoreColor(s.answer_relevance) }}>{fmtScore(s.answer_relevance)}</td>
                <td style={{ color: scoreColor(s.context_relevance) }}>{fmtScore(s.context_relevance)}</td>
                <td>{c.trace_id ? <code title={c.trace_id}>{c.trace_id.slice(0, 8)}</code> : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
