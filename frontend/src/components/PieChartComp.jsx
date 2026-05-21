// Pie / donut chart for share-of-whole visualizations.
// Donut mode is just a pie with an innerRadius — same data shape.

import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend,
} from 'recharts'

const PALETTE = [
  '#4f8cff', '#3fb950', '#d29922', '#bb86fc',
  '#f85149', '#56d4dd', '#ff7eb6', '#8be9fd',
]

export default function PieChartComp({ title, data, donut = false }) {
  const labels = data?.labels || []
  const values = (data?.values || []).map((v) => Number(v) || 0)
  const slices = labels.map((label, i) => ({ name: String(label), value: values[i] || 0 }))
  const total = values.reduce((s, v) => s + (Number.isFinite(v) ? v : 0), 0)

  // Defensive empty-state: a chart with no data is more confusing than no
  // chart at all. Show a labelled placeholder so the user understands the
  // chart was emitted but had no values to draw.
  if (slices.length === 0 || total === 0) {
    return (
      <div className="card">
        <div className="card-title">{title}</div>
        <div style={{
          color: 'var(--muted)', fontSize: 13, padding: '24px 8px',
          textAlign: 'center', border: '1px dashed var(--border)',
          borderRadius: 8,
        }}>
          No data to plot.
          {slices.length > 0 && total === 0 && ' All values are zero.'}
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={slices}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={90}
              innerRadius={donut ? 50 : 0}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              labelLine={false}
            >
              {slices.map((_, i) => (
                <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: '#161b22', border: '1px solid #2a313c' }}
              labelStyle={{ color: '#8b949e' }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
