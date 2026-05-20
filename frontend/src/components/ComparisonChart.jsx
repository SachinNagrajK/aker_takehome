// Grouped bar chart for comparing 2+ entities (units within a property OR
// 2+ properties) on one or more dimensions.

import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts'

const PALETTE = ['#4f8cff', '#3fb950', '#d29922', '#bb86fc', '#f85149', '#56d4dd']

export default function ComparisonChart({ title, data }) {
  const categories = data?.categories || []
  const rows = data?.rows || []
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
            <XAxis dataKey="dimension" stroke="#8b949e" fontSize={11} />
            <YAxis stroke="#8b949e" fontSize={11} />
            <Tooltip
              contentStyle={{ background: '#161b22', border: '1px solid #2a313c' }}
              labelStyle={{ color: '#8b949e' }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {categories.map((cat, i) => (
              <Bar key={cat} dataKey={cat} fill={PALETTE[i % PALETTE.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
