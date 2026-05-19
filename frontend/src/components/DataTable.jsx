export default function DataTable({ title, data }) {
  const columns = data?.columns || []
  const rows = data?.rows || []
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div style={{ maxHeight: 360, overflow: 'auto' }}>
        <table className="dtable">
          <thead>
            <tr>{columns.map((c, i) => <th key={i}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci}>{cell == null ? '—' : String(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
