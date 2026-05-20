import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="msg user">
        <div className="role">You</div>
        <div>{msg.content}</div>
      </div>
    )
  }
  if (msg.role === 'thinking') {
    return (
      <div className="msg assistant">
        <div className="role">Assistant</div>
        <div className="thinking">Thinking</div>
      </div>
    )
  }
  if (msg.role === 'error') {
    return (
      <div className="msg assistant">
        <div className="role">Error</div>
        <div className="error">{msg.content}</div>
      </div>
    )
  }
  // assistant
  return (
    <div className="msg assistant">
      <div className="role">Assistant</div>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
      </div>
      {msg.meta && (
        <div className="meta">
          {msg.meta.route && <span className="tag">route: {msg.meta.route}</span>}
          {msg.meta.llm && (
            <span className="tag">
              llm: {msg.meta.llm.provider}/{msg.meta.llm.model}
            </span>
          )}
          {msg.meta.scope_kind === 'compare' && msg.meta.property_codes?.length > 0 && (
            <span className="tag">compare: {msg.meta.property_codes.join(' ↔ ')}</span>
          )}
          {msg.meta.scope_kind === 'single' && msg.meta.property_code && msg.meta.scope_source && (
            <span className={`tag ${msg.meta.scope_source === 'resumed' ? 'scope-resumed' : ''}`}>
              scope: {msg.meta.property_code} ({msg.meta.scope_source})
            </span>
          )}
          {msg.meta.scope_enforced && <span className="tag">scope ✓</span>}
          {msg.meta.gave_up && <span className="tag gave-up">gave up</span>}
        </div>
      )}
    </div>
  )
}
