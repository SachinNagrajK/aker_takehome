export default function KPICard({ title, data }) {
  return (
    <div className="kpi">
      <div className="kpi-title">{title}</div>
      <div className="kpi-value">{data?.value ?? '—'}</div>
      {data?.subtitle && <div className="kpi-subtitle">{data.subtitle}</div>}
    </div>
  )
}
