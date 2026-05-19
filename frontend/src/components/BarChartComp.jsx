import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
} from 'recharts'

export default function BarChartComp({ title, data }) {
  const x = data?.x || []
  const y = data?.y || []
  const series = x.map((label, i) => ({ x: label, value: Number(y[i]) }))
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <BarChart data={series} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
            <XAxis dataKey="x" stroke="#8b949e" fontSize={11} />
            <YAxis stroke="#8b949e" fontSize={11} />
            <Tooltip
              contentStyle={{ background: '#161b22', border: '1px solid #2a313c' }}
              labelStyle={{ color: '#8b949e' }}
            />
            <Bar dataKey="value" fill="#4f8cff" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
