// Flattens (provider, models) into a single dropdown. Disables entries
// whose provider has `available: false` so users don't pick a key-less one.
export default function LLMSelector({ llms, value, onChange }) {
  const options = []
  for (const entry of llms) {
    for (const m of entry.models) {
      options.push({
        provider: entry.provider,
        model: m,
        label: `${entry.provider}/${m}${entry.available ? '' : ' (no key)'}`,
        available: entry.available,
      })
    }
  }
  const current = value || (options.find((o) => o.available)?.value)
  const key = (provider, model) => `${provider}::${model}`

  return (
    <div className="selector">
      <label>LLM</label>
      <select
        value={value ? key(value.provider, value.model) : ''}
        onChange={(e) => {
          const [provider, model] = e.target.value.split('::')
          onChange({ provider, model })
        }}
      >
        {options.map((o) => (
          <option
            key={key(o.provider, o.model)}
            value={key(o.provider, o.model)}
            disabled={!o.available}
          >
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}
