// Rendered when the LangGraph agent pauses at a clarification interrupt.
// Shows the agent's question and (if options provided) one-click reply buttons.
// Falls back to a freeform text input.

import { useState } from 'react'

export default function ClarificationCard({ clarification, onReply, disabled }) {
  const [freeform, setFreeform] = useState('')
  const question = clarification?.question || 'Please clarify.'
  const options = clarification?.options || []
  const kind = clarification?.scope_kind

  return (
    <div className="clarification-card">
      <div className="clarification-title">
        Clarification needed{kind ? ` · ${kind}` : ''}
      </div>
      <div className="clarification-question">{question}</div>

      {options.length > 0 && (
        <div className="clarification-options">
          {options.map((opt) => (
            <button
              key={opt}
              className="clarification-option"
              disabled={disabled}
              onClick={() => onReply(opt)}
            >
              {opt}
            </button>
          ))}
        </div>
      )}

      <form
        className="clarification-freeform"
        onSubmit={(e) => {
          e.preventDefault()
          if (freeform.trim()) onReply(freeform.trim())
        }}
      >
        <input
          type="text"
          placeholder={
            options.length > 0
              ? 'Or type your own (comma-separated for compare)…'
              : 'Type the property code (or comma-separated codes)…'
          }
          value={freeform}
          onChange={(e) => setFreeform(e.target.value)}
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || !freeform.trim()}>Send</button>
      </form>
    </div>
  )
}
