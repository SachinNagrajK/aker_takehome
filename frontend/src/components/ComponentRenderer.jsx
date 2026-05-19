// Dynamic dispatcher — picks the right visual based on `type`. Adding a new
// component type is a single switch entry plus its file.
import KPICard from './KPICard.jsx'
import DataTable from './DataTable.jsx'
import LineChartComp from './LineChartComp.jsx'
import BarChartComp from './BarChartComp.jsx'

export default function ComponentRenderer({ component, index }) {
  const { type, title, data } = component
  switch (type) {
    case 'kpi':         return <KPICard title={title} data={data} />
    case 'table':       return <DataTable title={title} data={data} />
    case 'line_chart':  return <LineChartComp title={title} data={data} />
    case 'bar_chart':   return <BarChartComp title={title} data={data} />
    default:
      return (
        <div className="card">
          <div className="card-title">Unknown component: {type}</div>
          <pre style={{ fontSize: 11, overflow: 'auto' }}>
            {JSON.stringify({ title, data }, null, 2)}
          </pre>
        </div>
      )
  }
}
