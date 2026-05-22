// Renders an image surfaced by RAG v2 (floor plans, amenity photos, etc.).
// `data` shape: { src, caption, source_url }
export default function ImageCard({ title, data }) {
  const { src, caption, source_url } = data || {}
  if (!src) return null
  return (
    <figure className="card image-card" style={{ margin: 0 }}>
      {title ? <div className="card-title">{title}</div> : null}
      <img
        src={src}
        alt={caption || title || 'image'}
        loading="lazy"
        style={{ maxWidth: '100%', height: 'auto', borderRadius: 8, display: 'block' }}
      />
      {(caption || source_url) ? (
        <figcaption style={{ fontSize: 12, opacity: 0.75, marginTop: 6 }}>
          {caption}
          {source_url ? (
            <>
              {caption ? ' — ' : ''}
              <a href={source_url} target="_blank" rel="noreferrer">source</a>
            </>
          ) : null}
        </figcaption>
      ) : null}
    </figure>
  )
}
