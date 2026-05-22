// Dispatcher — picks the right visual based on `type`.
// `image` types are collected by Message.jsx and rendered together via
// ImageGallery rather than one-card-per-image.
import KPICard from './KPICard.jsx'
import DataTable from './DataTable.jsx'
import LineChartComp from './LineChartComp.jsx'
import BarChartComp from './BarChartComp.jsx'
import ComparisonChart from './ComparisonChart.jsx'
import PieChartComp from './PieChartComp.jsx'

export default function ComponentRenderer({ component }) {
  const { type, title, data } = component
  switch (type) {
    case 'kpi':              return <KPICard title={title} data={data} />
    case 'table':            return <DataTable title={title} data={data} />
    case 'line_chart':       return <LineChartComp title={title} data={data} />
    case 'bar_chart':        return <BarChartComp title={title} data={data} />
    case 'comparison_chart': return <ComparisonChart title={title} data={data} />
    case 'pie_chart':        return <PieChartComp title={title} data={data} />
    case 'donut_chart':      return <PieChartComp title={title} data={data} donut />
    default:
      return (
        <div className="card">
          <div className="card-title">Unknown component: {type}</div>
          <pre style={{ fontSize: 11, overflow: 'auto', color: 'var(--muted)' }}>
            {JSON.stringify({ title, data }, null, 2)}
          </pre>
        </div>
      )
  }
}
