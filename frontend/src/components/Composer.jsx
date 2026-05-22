import { useState } from 'react'
import { ArrowUp } from 'lucide-react'

export default function Composer({ disabled, onSend }) {
  const [text, setText] = useState('')

  function send() {
    const v = text.trim()
    if (!v || disabled) return
    onSend(v)
    setText('')
  }

  return (
    <div className="composer">
      <div className="composer-input-wrap">
        <textarea
          placeholder="Ask about rent, leases, amenities, photos…  (Enter to send · Shift+Enter for newline)"
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
        />
      </div>
      <button
        className="send-btn"
        onClick={send}
        disabled={disabled || !text.trim()}
        aria-label="Send"
        title="Send (Enter)"
      >
        <ArrowUp size={18} />
      </button>
    </div>
  )
}
