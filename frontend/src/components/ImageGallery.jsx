// Inline image gallery for RAG v2 results. Renders thumbnails in a grid;
// click opens the shared Lightbox.
import { useState } from 'react'
import { ImageIcon } from 'lucide-react'
import Lightbox from './Lightbox.jsx'

export default function ImageGallery({ images }) {
  const [active, setActive] = useState(null)
  if (!images || images.length === 0) return null

  return (
    <>
      <div className="image-gallery">
        <div className="image-gallery-header">
          <ImageIcon size={12} />
          <span>{images.length} image{images.length === 1 ? '' : 's'} from the property</span>
        </div>
        <div className="image-grid">
          {images.map((img, i) => (
            <div
              key={i}
              className="image-thumb"
              onClick={() => setActive(img)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter') setActive(img) }}
            >
              <img src={img.src} alt={img.caption || `image ${i + 1}`} loading="lazy" />
              {img.caption && (
                <div className="image-thumb-caption">{img.caption}</div>
              )}
            </div>
          ))}
        </div>
      </div>
      <Lightbox image={active} onClose={() => setActive(null)} />
    </>
  )
}
