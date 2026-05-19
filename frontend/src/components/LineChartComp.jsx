import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts'

export default function LineChartComp({ title, data }) {
  const x = data?.x || []
  const y = data?.y || []
  const secondary = data?.secondary
  const series = x.map((label, i) => {
    const point = { x: label, primary: Number(y[i]) }
    if (secondary && secondary.y) point.secondary = Number(secondary.y[i])
    return point
  })

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={series} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
            <XAxis dataKey="x" stroke="#8b949e" fontSize={11} />
            <YAxis stroke="#8b949e" fontSize={11} />
            <Tooltip
              contentStyle={{ background: '#161b22', border: '1px solid #2a313c' }}
              labelStyle={{ color: '#8b949e' }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone" dataKey="primary"
              name={data?.y_label || 'value'} stroke="#4f8cff"
              dot={{ r: 2 }} strokeWidth={2}
            />
            {secondary && (
              <Line
                type="monotone" dataKey="secondary"
                name={secondary.label} stroke="#3fb950"
                dot={{ r: 2 }} strokeWidth={2}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
