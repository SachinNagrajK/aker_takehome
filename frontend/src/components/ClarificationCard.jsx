// Inline clarification panel — surfaces when the agent paused at an
// interrupt (in v3 only the "missing" scope case, since dropdown+message
// disagreement now auto-promotes to compare mode).
import { useState } from 'react'
import { HelpCircle } from 'lucide-react'
import { motion } from 'framer-motion'

export default function ClarificationCard({ clarification, onReply, disabled }) {
  const [freeform, setFreeform] = useState('')
  const question = clarification?.question || 'Please clarify.'
  const options = clarification?.options || []
  const kind = clarification?.scope_kind

  const tooManyOptions = options.length > 8
  const visibleOptions = tooManyOptions ? options.slice(0, 8) : options

  return (
    <motion.div
      className="clarification-card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <div className="clarification-title">
        <HelpCircle size={12} />
        Clarification needed{kind ? ` · ${kind}` : ''}
      </div>
      <div className="clarification-question">{question}</div>

      {visibleOptions.length > 0 && (
        <div className="clarification-options">
          {visibleOptions.map((opt) => (
            <button
              key={opt}
              className="clarification-option"
              disabled={disabled}
              onClick={() => onReply(opt)}
            >
              {opt}
            </button>
          ))}
          {tooManyOptions && (
            <span className="clarification-more">
              + {options.length - visibleOptions.length} more — type one below
            </span>
          )}
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
              ? 'Or type your own — multiple codes welcome (e.g. "compare 115r and 134r")'
              : 'Type a property code or "compare X and Y"…'
          }
          value={freeform}
          onChange={(e) => setFreeform(e.target.value)}
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || !freeform.trim()}>Send</button>
      </form>
    </motion.div>
  )
}
