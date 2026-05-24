import { useState } from 'react'
import { ArrowUp, Square } from 'lucide-react'

export default function Composer({ disabled, busy, onSend, onStop }) {
  const [text, setText] = useState('')

  function send() {
    const v = text.trim()
    if (!v || disabled) return
    onSend(v)
    setText('')
  }

  // When the assistant is generating, morph the send button into a stop
  // button (like Claude / ChatGPT). Clicking it aborts the in-flight stream
  // via the AbortController owned by App.jsx.
  const showStop = busy && typeof onStop === 'function'

  return (
    <div className="composer">
      <div className="composer-input-wrap">
        <textarea
          placeholder="Ask about rent, leases, amenities, photos…  (Enter to send · Shift+Enter for newline)"
          value={text}
          disabled={disabled && !showStop}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
        />
      </div>
      {showStop ? (
        <button
          className="send-btn stop-btn"
          onClick={onStop}
          aria-label="Stop generating"
          title="Stop generating"
        >
          <Square size={14} fill="currentColor" />
        </button>
      ) : (
        <button
          className="send-btn"
          onClick={send}
          disabled={disabled || !text.trim()}
          aria-label="Send"
          title="Send (Enter)"
        >
          <ArrowUp size={18} />
        </button>
      )}
    </div>
  )
}
