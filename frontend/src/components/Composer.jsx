import { useState } from 'react'

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
      <textarea
        placeholder="Ask anything about this property... (e.g. 'What is the average rent?')"
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
      <button onClick={send} disabled={disabled || !text.trim()}>Send</button>
    </div>
  )
}
