import { useEffect, useMemo, useRef, useState } from 'react'
import { Building2 } from 'lucide-react'
import { api } from './api.js'
import PropertySelector from './components/PropertySelector.jsx'
import LLMSelector from './components/LLMSelector.jsx'
import Composer from './components/Composer.jsx'
import Message from './components/Message.jsx'
import ClarificationCard from './components/ClarificationCard.jsx'
import EmptyState from './components/EmptyState.jsx'

function genConversationId() {
  return (crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`)
}

const SUGGESTIONS = [
  'What is the average rent and occupancy?',
  'Show me the unit mix breakdown.',
  'How has the average rent changed over the year?',
  'Which leases are expiring in the next 90 days?',
  'Show me the gallery and amenities',
  'Which units have the highest outstanding balance?',
]

export default function App() {
  const [properties, setProperties] = useState([])
  const [llms, setLlms] = useState([])
  // v3: propertyCodes is an ARRAY now to support compare mode from the dropdown.
  const [propertyCodes, setPropertyCodes] = useState([])
  const [llm, setLlm] = useState(null)
  const [messages, setMessages] = useState([])
  const [pendingClarification, setPendingClarification] = useState(null)
  const [conversationId, setConversationId] = useState(genConversationId())
  const [lastUserMessage, setLastUserMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [bootError, setBootError] = useState(null)
  const scrollRef = useRef(null)
  // AbortController for the in-flight chat stream — lets the user click the
  // composer's Stop button (rendered when busy=true) to cancel generation.
  const abortRef = useRef(null)

  useEffect(() => {
    (async () => {
      try {
        const [props, llmList] = await Promise.all([api.properties(), api.llms()])
        setProperties(props)
        setLlms(llmList)
        if (props.length) setPropertyCodes([props[0].property_code])
        const firstAvail = llmList.find((l) => l.available)
        if (firstAvail) setLlm({ provider: firstAvail.provider, model: firstAvail.models[0] })
      } catch (e) {
        setBootError(e.message)
      }
    })()
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  // Whenever the dropdown scope changes (add/remove a code) → fresh conversation.
  const propertyCodesKey = propertyCodes.join(',')
  useEffect(() => {
    setMessages([])
    setPendingClarification(null)
    setConversationId(genConversationId())
  }, [propertyCodesKey])

  // For the header pill we summarise the active selection.
  const activeProperty = useMemo(
    () => (propertyCodes.length === 1
      ? properties.find((p) => p.property_code === propertyCodes[0])
      : null),
    [properties, propertyCodesKey],
  )

  // Stream a single chat turn. Reasoning lines accumulate in the assistant
  // message's `progress` array while tokens fill `content`. When `done`
  // fires we replace the streaming bubble with the final structured response.
  function handleStop() {
    // Abort the in-flight fetch — the catch block in streamChat handles the
    // AbortError quietly and finalises the streaming bubble as "stopped".
    if (abortRef.current) {
      try { abortRef.current.abort() } catch { /* noop */ }
    }
  }

  async function streamChat(payload) {
    setBusy(true)
    // Fresh AbortController per turn so previous aborts don't bleed into new ones.
    const controller = new AbortController()
    abortRef.current = controller
    // Append a streaming-placeholder assistant message; we'll mutate it in place.
    setMessages((m) => [
      ...m,
      { role: 'assistant_streaming', content: '', progress: [], meta: {} },
    ])

    const updateLast = (mutator) => {
      setMessages((m) => {
        if (m.length === 0) return m
        const next = m.slice()
        const last = next[next.length - 1]
        if (last.role !== 'assistant_streaming') return m
        next[next.length - 1] = mutator(last)
        return next
      })
    }

    const setBubbleError = (msg) => {
      setMessages((m) => {
        if (m.length === 0) return m
        const next = m.slice()
        // If the last bubble is the streaming one, replace it; otherwise append.
        if (next[next.length - 1].role === 'assistant_streaming') {
          next[next.length - 1] = { role: 'error', content: msg }
        } else {
          next.push({ role: 'error', content: msg })
        }
        return next
      })
    }

    try {
      await api.chatStream(payload, {
        signal: controller.signal,
        onEvent: (evt) => {
          switch (evt.type) {
            case 'open':
              if (evt.conversation_id) setConversationId(evt.conversation_id)
              break
            case 'step':
              updateLast((b) => ({
                ...b,
                progress: [...b.progress, {
                  kind: 'step', text: evt.message, node: evt.node, status: 'ok',
                }],
              }))
              break
            case 'tool':
              updateLast((b) => ({
                ...b,
                progress: [...b.progress, {
                  kind: 'tool', text: evt.reasoning, tool: evt.tool, status: 'running',
                }],
              }))
              break
            case 'tool_end':
              updateLast((b) => {
                const prog = b.progress.slice()
                // Find the most recent running entry for this tool name.
                for (let i = prog.length - 1; i >= 0; i--) {
                  if (prog[i].kind === 'tool' && prog[i].tool === evt.tool && prog[i].status === 'running') {
                    prog[i] = {
                      ...prog[i],
                      status: evt.ok ? 'ok' : 'err',
                      duration_ms: evt.duration_ms,
                      error: evt.error,
                    }
                    break
                  }
                }
                return { ...b, progress: prog }
              })
              break
            case 'delta':
              updateLast((b) => ({ ...b, content: (b.content || '') + evt.text }))
              break
            case 'clarification':
              setPendingClarification(evt.payload)
              // Pop the streaming bubble — clarification card replaces it.
              setMessages((m) => {
                if (m.length === 0) return m
                if (m[m.length - 1].role === 'assistant_streaming') return m.slice(0, -1)
                return m
              })
              break
            case 'done': {
              const r = evt.response || {}
              // Stream ended — coerce any still-running progress entry to ok
              // so no spinner is left frozen.
              updateLast((b) => {
                const finalProgress = (b.progress || []).map((p) =>
                  p.status === 'running' ? { ...p, status: 'ok' } : p
                )
                return {
                  role: 'assistant',
                  content: r.answer_markdown ?? b.content,
                  meta: {
                    route: r.route,
                    llm: r.llm,
                    scope_enforced: r.scope_enforced,
                    scope_kind: r.scope_kind,
                    scope_source: r.scope_source,
                    property_code: r.property_code,
                    property_codes: r.property_codes,
                    gave_up: r.gave_up,
                    components: r.components || [],
                    sources: r.sources || [],
                    tool_trace: r.tool_trace || [],
                    progress: finalProgress,
                  },
                }
              })
              if (r.conversation_id) setConversationId(r.conversation_id)
              break
            }
            case 'error':
              setBubbleError(evt.message || 'Stream error')
              break
            default:
              break
          }
        },
      })
    } catch (e) {
      // User clicked Stop → finalise whatever streamed and add a "stopped"
      // marker. AbortError can surface under a few different names depending
      // on the fetch implementation; check all common ones.
      const aborted = e?.name === 'AbortError' || e?.code === 20 ||
                      controller.signal.aborted
      if (aborted) {
        updateLast((b) => ({
          role: 'assistant',
          content: b.content || '_(stopped)_',
          meta: {
            ...(b.meta || {}),
            stopped: true,
            progress: (b.progress || []).map((p) =>
              p.status === 'running' ? { ...p, status: 'stopped' } : p
            ),
          },
        }))
      } else {
        setBubbleError(e.message)
      }
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  async function handleSend(text) {
    if (!llm) return
    setLastUserMessage(text)
    setMessages((m) => [...m, { role: 'user', content: text }])
    await streamChat({
      property_code: propertyCodes.length === 1 ? propertyCodes[0] : (propertyCodes.length > 0 ? propertyCodes : null),
      message: text,
      llm_provider: llm.provider,
      model: llm.model,
      conversation_id: conversationId,
    })
  }

  async function handleClarificationReply(choice) {
    if (!choice || !llm) return
    setPendingClarification(null)
    setMessages((m) => [...m, { role: 'user', content: `(chose ${choice})` }])
    await streamChat({
      property_code: propertyCodes.length === 1 ? propertyCodes[0] : (propertyCodes.length > 0 ? propertyCodes : null),
      message: lastUserMessage,
      llm_provider: llm.provider,
      model: llm.model,
      conversation_id: conversationId,
      clarification_reply: choice,
    })
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
        <div className="brand">
          <span className="brand-icon"><Building2 size={16} /></span>
          Property AI
        </div>
        <PropertySelector
          properties={properties}
          value={propertyCodes}
          onChange={setPropertyCodes}
        />
        <LLMSelector llms={llms} value={llm} onChange={setLlm} />
        {propertyCodes.length === 1 && activeProperty && (
          <div className="scope-pill">
            <span>Scope</span>
            <strong>{activeProperty.property_code}</strong>
            <span style={{ opacity: 0.7 }}>·</span>
            <span>{activeProperty.property_name}</span>
          </div>
        )}
        {propertyCodes.length > 1 && (
          <div className="scope-pill" title={propertyCodes.join(', ')}>
            <span>Compare</span>
            <strong>{propertyCodes.length} properties</strong>
            <span style={{ opacity: 0.7 }}>·</span>
            <span>{propertyCodes.join(' ↔ ')}</span>
          </div>
        )}
      </div>

      <div className="main">
        <div className="chat">
          <div className="messages" ref={scrollRef}>
            <div className="messages-inner">
              {messages.length === 0 && (
                <EmptyState
                  propertyName={activeProperty?.property_name || null}
                  disabled={busy || !llm}
                  onPick={handleSend}
                />
              )}
              {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            </div>
          </div>

          <div className="composer-wrap">
            {pendingClarification && (
              <ClarificationCard
                clarification={pendingClarification}
                onReply={handleClarificationReply}
                disabled={busy}
              />
            )}

            {/* Suggestion chips moved into <EmptyState/> hero grid above.
                Original chip strip preserved below for easy restore:
                {!pendingClarification && messages.length === 0 && (
                  <div className="suggestions">
                    {SUGGESTIONS.map((s) => (
                      <button key={s} className="chip" disabled={busy}
                        onClick={() => handleSend(s)}>{s}</button>
                    ))}
                  </div>
                )} */}

            <Composer
              disabled={busy || !!pendingClarification}
              busy={busy}
              onSend={handleSend}
              onStop={handleStop}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
