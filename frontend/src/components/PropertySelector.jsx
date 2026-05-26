// Single-select property picker. The UI mirrors the LLM selector to its right:
// a single styled dropdown that always holds exactly ONE property code (or none
// before the user has picked). No add/remove chips — selecting any item
// replaces the current selection.
import { useState, useRef, useEffect } from 'react'
import { ChevronDown, Building2, Check } from 'lucide-react'

export default function PropertySelector({ properties, value, onChange }) {
  // `value` from the parent is a string[] for backward compat; we only ever
  // emit a single-element array so the rest of the app keeps working unchanged.
  const codes = Array.isArray(value) ? value : (value ? [value] : [])
  const selectedCode = codes[0] || ''
  const selected = properties.find((p) => p.property_code === selectedCode) || null

  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const wrapperRef = useRef(null)

  useEffect(() => {
    function onDoc(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false)
        setFilter('')
      }
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  function pick(code) {
    onChange([code])
    setOpen(false)
    setFilter('')
  }

  const filtered = properties.filter((p) => {
    if (!filter.trim()) return true
    const f = filter.toLowerCase()
    return (
      p.property_code.toLowerCase().includes(f) ||
      p.property_name.toLowerCase().includes(f)
    )
  })

  return (
    <div className="selector property-select" ref={wrapperRef}>
      <label>Property</label>
      <button
        type="button"
        className="property-select-trigger"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <Building2 size={13} className="property-select-icon" />
        {selected ? (
          <span className="property-select-label">
            <span className="property-select-code">{selected.property_code}</span>
            <span className="property-select-name">{selected.property_name}</span>
          </span>
        ) : (
          <span className="property-select-placeholder">Pick a property…</span>
        )}
        <ChevronDown size={14} className="property-select-caret" />
      </button>

      {open && (
        <div className="property-menu" role="listbox">
          <input
            autoFocus
            type="text"
            className="property-menu-search"
            placeholder="Filter by code or name…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && filtered.length > 0) {
                pick(filtered[0].property_code)
              }
              if (e.key === 'Escape') {
                setOpen(false)
                setFilter('')
              }
            }}
          />
          <div className="property-menu-list">
            {filtered.length === 0 ? (
              <div className="property-menu-empty">No matches.</div>
            ) : (
              filtered.map((p) => {
                const isSelected = p.property_code === selectedCode
                return (
                  <button
                    key={p.property_code}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    className={`property-menu-item${isSelected ? ' is-selected' : ''}`}
                    onClick={() => pick(p.property_code)}
                  >
                    <span className="property-menu-code">{p.property_code}</span>
                    <span className="property-menu-name">{p.property_name}</span>
                    <span className="property-menu-type">{p.property_type}</span>
                    {isSelected && <Check size={12} className="property-menu-check" />}
                  </button>
                )
              })
            )}
          </div>
        </div>
      )}
    </div>
  )
}
