// Multi-select property picker. Chips show currently active codes —
// click × to remove. Picking 2+ properties puts the scope into compare mode.
import { useState, useRef, useEffect } from 'react'
import { X, Plus, Building2 } from 'lucide-react'

export default function PropertySelector({ properties, value, onChange }) {
  // `value` is a string[] of currently selected property codes.
  const codes = Array.isArray(value) ? value : (value ? [value] : [])
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

  function add(code) {
    if (!codes.includes(code)) onChange([...codes, code])
    setOpen(false)
    setFilter('')
  }
  function remove(code) {
    onChange(codes.filter((c) => c !== code))
  }

  const remaining = properties
    .filter((p) => !codes.includes(p.property_code))
    .filter((p) => {
      if (!filter.trim()) return true
      const f = filter.toLowerCase()
      return (
        p.property_code.toLowerCase().includes(f) ||
        p.property_name.toLowerCase().includes(f)
      )
    })

  return (
    <div className="selector property-multi" ref={wrapperRef}>
      <label>
        Property{codes.length > 1 ? ` · compare ${codes.length}` : ''}
      </label>
      <div className="property-chips">
        {codes.map((c) => {
          const p = properties.find((pp) => pp.property_code === c)
          return (
            <span key={c} className="property-chip" title={p?.property_name || c}>
              <Building2 size={11} />
              <span className="property-chip-code">{c}</span>
              {p?.property_name && (
                <span className="property-chip-name">{p.property_name}</span>
              )}
              <button
                type="button"
                onClick={() => remove(c)}
                aria-label={`Remove ${c}`}
                className="property-chip-x"
              >
                <X size={12} />
              </button>
            </span>
          )
        })}
        <button
          type="button"
          className="property-add"
          onClick={() => setOpen((o) => !o)}
          disabled={remaining.length === 0 && !filter}
          aria-label="Add property"
        >
          <Plus size={13} />
          <span>{codes.length === 0 ? 'Pick property' : 'Add'}</span>
        </button>
        {open && (
          <div className="property-menu">
            <input
              autoFocus
              type="text"
              className="property-menu-search"
              placeholder="Filter by code or name…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && remaining.length > 0) {
                  add(remaining[0].property_code)
                }
                if (e.key === 'Escape') {
                  setOpen(false)
                  setFilter('')
                }
              }}
            />
            <div className="property-menu-list">
              {remaining.length === 0 ? (
                <div className="property-menu-empty">No matches.</div>
              ) : (
                remaining.map((p) => (
                  <button
                    key={p.property_code}
                    type="button"
                    className="property-menu-item"
                    onClick={() => add(p.property_code)}
                  >
                    <span className="property-menu-code">{p.property_code}</span>
                    <span className="property-menu-name">{p.property_name}</span>
                    <span className="property-menu-type">{p.property_type}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
