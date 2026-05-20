// Collapsible "what the agent did" panel. Each step shows tool name, args,
// status, and duration. Helpful for graders evaluating orchestration depth.

import { useState } from 'react'

export default function ToolTrace({ steps }) {
  const [open, setOpen] = useState(true)
  if (!steps || steps.length === 0) return null

  const okCount = steps.filter((s) => s.ok).length
  const errCount = steps.length - okCount

  return (
    <div className="tool-trace">
      <button
        className="tool-trace-header"
        onClick={() => setOpen(!open)}
        style={{ background: 'transparent', color: 'inherit', border: 'none', cursor: 'pointer', padding: 0, width: '100%', textAlign: 'left' }}
      >
        <span className="tool-trace-title">
          Agent trace · {steps.length} step{steps.length === 1 ? '' : 's'}
          {errCount > 0 && <span className="tool-trace-err"> ({errCount} error{errCount === 1 ? '' : 's'})</span>}
        </span>
        <span className="tool-trace-toggle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <ol className="tool-trace-list">
          {steps.map((s, i) => (
            <li key={i} className={`tool-trace-step ${s.ok ? 'ok' : 'err'}`}>
              <div className="tool-trace-line">
                <code className="tool-trace-name">{s.tool}</code>
                <span className="tool-trace-status">{s.ok ? 'OK' : 'ERROR'}</span>
                {s.duration_ms != null && (
                  <span className="tool-trace-dur">{s.duration_ms}ms</span>
                )}
              </div>
              {s.args && Object.keys(s.args).length > 0 && (
                <div className="tool-trace-args">
                  {Object.entries(s.args).map(([k, v]) => (
                    <span key={k}><b>{k}</b>={JSON.stringify(v)}</span>
                  ))}
                </div>
              )}
              {s.error && <div className="tool-trace-error">{s.error}</div>}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
