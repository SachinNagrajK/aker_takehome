import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import PropertySelector from './components/PropertySelector.jsx'
import LLMSelector from './components/LLMSelector.jsx'
import Composer from './components/Composer.jsx'
import Message from './components/Message.jsx'
import ComponentRenderer from './components/ComponentRenderer.jsx'
import ClarificationCard from './components/ClarificationCard.jsx'
import ToolTrace from './components/ToolTrace.jsx'

// Generate a per-conversation id so the backend can resume interrupted
// graph runs (LangGraph checkpointer thread_id).
function genConversationId() {
  return (crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`)
}

const SUGGESTIONS = [
  'What is the average rent and occupancy?',
  'Show me the unit mix breakdown.',
  'How has the average rent changed over the year?',
  'Which leases are expiring in the next 90 days?',
  'What amenities does this property offer?',
  'Which units have the highest outstanding balance?',
]

export default function App() {
  const [properties, setProperties] = useState([])
  const [llms, setLlms] = useState([])
  const [propertyCode, setPropertyCode] = useState('')
  const [llm, setLlm] = useState(null)
  const [messages, setMessages] = useState([])
  const [components, setComponents] = useState([])
  const [sources, setSources] = useState([])
  const [toolTrace, setToolTrace] = useState([])
  const [pendingClarification, setPendingClarification] = useState(null)
  const [conversationId, setConversationId] = useState(genConversationId())
  const [lastUserMessage, setLastUserMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [bootError, setBootError] = useState(null)
  const scrollRef = useRef(null)

  // Boot: fetch /properties and /llms in parallel.
  useEffect(() => {
    (async () => {
      try {
        const [props, llmList] = await Promise.all([api.properties(), api.llms()])
        setProperties(props)
        setLlms(llmList)
        if (props.length) setPropertyCode(props[0].property_code)
        const firstAvail = llmList.find((l) => l.available)
        if (firstAvail) setLlm({ provider: firstAvail.provider, model: firstAvail.models[0] })
      } catch (e) {
        setBootError(e.message)
      }
    })()
  }, [])

  // Auto-scroll messages to bottom when a new one arrives.
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  // Whenever property changes, clear the chat, components, and conversation.
  useEffect(() => {
    setMessages([])
    setComponents([])
    setSources([])
    setToolTrace([])
    setPendingClarification(null)
    setConversationId(genConversationId())
  }, [propertyCode])

  const activeProperty = useMemo(
    () => properties.find((p) => p.property_code === propertyCode),
    [properties, propertyCode],
  )

  function applyChatResponse(res, replaceLastThinking = true) {
    if (res.clarification) {
      // Paused at an interrupt — render the question, don't push a final reply.
      setPendingClarification(res.clarification)
      setMessages((m) => (replaceLastThinking ? m.slice(0, -1) : m))
      return
    }
    setPendingClarification(null)
    setMessages((m) => {
      const next = replaceLastThinking ? m.slice(0, -1) : m
      next.push({
        role: 'assistant',
        content: res.answer_markdown,
        meta: {
          route: res.route,
          llm: res.llm,
          scope_enforced: res.scope_enforced,
          scope_kind: res.scope_kind,
          scope_source: res.scope_source,
          property_code: res.property_code,
          property_codes: res.property_codes,
          gave_up: res.gave_up,
        },
      })
      return next
    })
    setComponents(res.components || [])
    setSources(res.sources || [])
    setToolTrace(res.tool_trace || [])
  }

  async function handleSend(text) {
    if (!llm) return
    setBusy(true)
    setLastUserMessage(text)
    setMessages((m) => [
      ...m,
      { role: 'user', content: text },
      { role: 'thinking' },
    ])
    try {
      const res = await api.chat({
        property_code: propertyCode || null,
        message: text,
        llm_provider: llm.provider,
        model: llm.model,
        conversation_id: conversationId,
      })
      applyChatResponse(res)
    } catch (e) {
      setMessages((m) => {
        const next = m.slice(0, -1)
        next.push({ role: 'error', content: e.message })
        return next
      })
    } finally {
      setBusy(false)
    }
  }

  async function handleClarificationReply(choice) {
    if (!choice || !llm) return
    setBusy(true)
    setPendingClarification(null)
    setMessages((m) => [
      ...m,
      { role: 'user', content: `(chose ${choice})` },
      { role: 'thinking' },
    ])
    try {
      const res = await api.chat({
        property_code: propertyCode || null,
        message: lastUserMessage,
        llm_provider: llm.provider,
        model: llm.model,
        conversation_id: conversationId,
        clarification_reply: choice,
      })
      applyChatResponse(res)
    } catch (e) {
      setMessages((m) => {
        const next = m.slice(0, -1)
        next.push({ role: 'error', content: e.message })
        return next
      })
    } finally {
      setBusy(false)
    }
  }

  if (bootError) {
    return (
      <div className="app">
        <div style={{ padding: 40 }}>
          <h2>Backend unreachable</h2>
          <div className="error">{bootError}</div>
          <p style={{ color: 'var(--muted)' }}>
            Start the FastAPI server: <code>uvicorn app.main:app --reload</code>
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">Property AI Assistant</div>
        <PropertySelector
          properties={properties}
          value={propertyCode}
          onChange={setPropertyCode}
        />
        <LLMSelector
          llms={llms}
          value={llm}
          onChange={setLlm}
        />
        {activeProperty && (
          <div className="scope-pill">
            Scope: <strong>{activeProperty.property_code}</strong> · {activeProperty.property_name}
          </div>
        )}
      </div>

      <div className="main">
        <div className="chat">
          <div className="messages" ref={scrollRef}>
            {messages.length === 0 && (
              <div style={{ color: 'var(--muted)', marginTop: '20vh', textAlign: 'center' }}>
                <p>Ask anything about <strong>{activeProperty?.property_name || '...'}</strong>.</p>
                <p style={{ fontSize: 12 }}>Pick a suggestion below or type your own question.</p>
              </div>
            )}
            {messages.map((msg, i) => <Message key={i} msg={msg} />)}
          </div>

          {pendingClarification && (
            <ClarificationCard
              clarification={pendingClarification}
              onReply={handleClarificationReply}
              disabled={busy}
            />
          )}

          {!pendingClarification && (
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="chip"
                  disabled={busy}
                  onClick={() => handleSend(s)}
                  style={{ background: 'transparent', color: 'inherit', padding: '6px 12px', border: '1px solid var(--border)' }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <Composer disabled={busy || !!pendingClarification} onSend={handleSend} />
        </div>

        <div className="components-pane">
          {components.length === 0 && sources.length === 0 && toolTrace.length === 0 ? (
            <div className="empty">Components, sources, and tool trace will appear here.</div>
          ) : (
            <>
              {components.map((c, i) => (
                <ComponentRenderer key={i} component={c} index={i} />
              ))}
              {sources.length > 0 && (
                <div className="sources">
                  <div className="source-title">Sources</div>
                  {sources.map((s, i) => (
                    <a key={i} href={s.url} target="_blank" rel="noreferrer">
                      {s.label}
                    </a>
                  ))}
                </div>
              )}
              {toolTrace.length > 0 && <ToolTrace steps={toolTrace} />}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
