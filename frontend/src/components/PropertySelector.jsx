export default function PropertySelector({ properties, value, onChange }) {
  return (
    <div className="selector">
      <label>Property</label>
      <select value={value || ''} onChange={(e) => onChange(e.target.value)}>
        {properties.map((p) => (
          <option key={p.property_code} value={p.property_code}>
            {p.property_code} — {p.property_name} ({p.property_type})
          </option>
        ))}
      </select>
    </div>
  )
}
