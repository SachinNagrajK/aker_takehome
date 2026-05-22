// One chat message. Assistant messages render markdown plus inline components,
// image gallery, sources strip, and the collapsible agent trace. The
// `assistant_streaming` role shows a live timeline of reasoning lines (one
// per node/tool) followed by the answer text as it streams in.
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion } from 'framer-motion'
import {
  Sparkles, User, AlertTriangle,
  Link as LinkIcon, ExternalLink,
  Loader2, Check, X as XIcon, Wrench,
} from 'lucide-react'

import ComponentRenderer from './ComponentRenderer.jsx'
import ImageGallery from './ImageGallery.jsx'
import ToolTrace from './ToolTrace.jsx'

function Avatar({ kind }) {
  if (kind === 'user') {
    return <div className="avatar"><User size={15} /></div>
  }
  if (kind === 'error') {
    return <div className="avatar"><AlertTriangle size={15} /></div>
  }
  return <div className="avatar"><Sparkles size={15} /></div>
}

function ProgressTimeline({ steps }) {
  if (!steps || steps.length === 0) return null
  return (
    <div className="progress-timeline">
      {steps.map((s, i) => {
        // Three render states: 'running' (spinner), 'ok'/'done' (check),
        // 'err' (X). Any other value gets coerced to 'done'.
        const raw = s.status || (s.kind === 'tool' ? 'running' : 'done')
        const status = (raw === 'running' || raw === 'err') ? raw : 'ok'

        let icon
        if (status === 'running') icon = <Loader2 size={12} className="spin" />
        else if (status === 'err') icon = <XIcon size={12} />
        else icon = <Check size={12} />

        return (
          <motion.div
            key={i}
            className={`progress-step status-${status}`}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.15 }}
          >
            <span className="progress-icon">{icon}</span>
            <span className="progress-text">{s.text}</span>
            {s.duration_ms != null && status !== 'running' && (
              <span className="progress-duration">{s.duration_ms}ms</span>
            )}
          </motion.div>
        )
      })}
    </div>
  )
}

function StreamingCaret() {
  return <span className="streaming-caret" aria-hidden="true" />
}

function SourcesStrip({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <div className="sources-strip">
      <span className="sources-label">
        <LinkIcon size={11} style={{ display: 'inline', verticalAlign: -1, marginRight: 4 }} />
        Sources
      </span>
      {sources.map((s, i) => (
        <a key={i} className="source-chip" href={s.url} target="_blank" rel="noreferrer">
          {s.label}
          <ExternalLink size={11} />
        </a>
      ))}
    </div>
  )
}

function MetaTags({ meta }) {
  if (!meta) return null
  return (
    <div className="meta">
      {meta.route && <span className="tag">route: {meta.route}</span>}
      {meta.llm && (
        <span className="tag">{meta.llm.provider}/{meta.llm.model}</span>
      )}
      {meta.scope_kind === 'compare' && meta.property_codes?.length > 0 && (
        <span className="tag">compare: {meta.property_codes.join(' ↔ ')}</span>
      )}
      {meta.scope_kind === 'single' && meta.property_code && meta.scope_source && (
        <span className={`tag ${meta.scope_source === 'resumed' ? 'scope-resumed' : ''}`}>
          scope: {meta.property_code} ({meta.scope_source})
        </span>
      )}
      {meta.scope_enforced && <span className="tag">scope ✓</span>}
      {meta.gave_up && <span className="tag gave-up">gave up</span>}
    </div>
  )
}

export default function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <motion.div
        className="msg user"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <Avatar kind="user" />
        <div className="msg-body">
          <div>{msg.content}</div>
        </div>
      </motion.div>
    )
  }

  if (msg.role === 'thinking') {
    return (
      <motion.div
        className="msg assistant"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
      >
        <Avatar kind="assistant" />
        <div className="msg-body">
          <div className="thinking">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span style={{ marginLeft: 4 }}>Thinking</span>
          </div>
        </div>
      </motion.div>
    )
  }

  if (msg.role === 'assistant_streaming') {
    const text = msg.content || ''
    return (
      <motion.div
        className="msg assistant"
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <Avatar kind="assistant" />
        <div className="msg-body">
          <ProgressTimeline steps={msg.progress} />
          {text ? (
            <div className="markdown streaming">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
              <StreamingCaret />
            </div>
          ) : (
            (msg.progress?.length ?? 0) === 0 && (
              <div className="thinking">
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span style={{ marginLeft: 4 }}>Thinking</span>
              </div>
            )
          )}
        </div>
      </motion.div>
    )
  }

  if (msg.role === 'error') {
    return (
      <motion.div
        className="msg error"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Avatar kind="error" />
        <div className="msg-body">
          <div className="error">{msg.content}</div>
        </div>
      </motion.div>
    )
  }

  // Assistant final message — split image components into a gallery and pass
  // everything else through the regular renderer so charts/KPIs/tables flow
  // inline beneath the markdown.
  const components = msg.meta?.components || []
  const imageComponents = components.filter((c) => c.type === 'image')
  const otherComponents = components.filter((c) => c.type !== 'image')
  const images = imageComponents.map((c) => ({
    src: c.data?.src,
    caption: c.data?.caption || c.title,
    source_url: c.data?.source_url,
  })).filter((i) => i.src)

  const progress = msg.meta?.progress || []

  return (
    <motion.div
      className="msg assistant"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
    >
      <Avatar kind="assistant" />
      <div className="msg-body">
        {progress.length > 0 && (
          <details className="progress-collapsed">
            <summary>
              <Wrench size={11} /> Reasoning · {progress.length} step{progress.length === 1 ? '' : 's'}
            </summary>
            <ProgressTimeline steps={progress} />
          </details>
        )}

        <div className="markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content || ''}</ReactMarkdown>
        </div>

        {(images.length > 0 || otherComponents.length > 0 || (msg.meta?.sources?.length > 0) || (msg.meta?.tool_trace?.length > 0)) && (
          <div className="msg-components">
            {otherComponents.map((c, i) => (
              <ComponentRenderer key={i} component={c} />
            ))}
            {images.length > 0 && <ImageGallery images={images} />}
            {msg.meta?.sources?.length > 0 && (
              <SourcesStrip sources={msg.meta.sources} />
            )}
            {msg.meta?.tool_trace?.length > 0 && (
              <ToolTrace steps={msg.meta.tool_trace} />
            )}
          </div>
        )}

        <MetaTags meta={msg.meta} />
      </div>
    </motion.div>
  )
}
