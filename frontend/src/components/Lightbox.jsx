// Fullscreen image viewer. Click overlay or X to close. Esc also closes.
import { useEffect } from 'react'
import { X, ExternalLink } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export default function Lightbox({ image, onClose }) {
  useEffect(() => {
    if (!image) return
    function onKey(e) { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [image, onClose])

  return (
    <AnimatePresence>
      {image && (
        <motion.div
          className="lightbox-overlay"
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <button className="lightbox-close" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
          <motion.div
            className="lightbox-content"
            onClick={(e) => e.stopPropagation()}
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
          >
            <img src={image.src} alt={image.caption || 'image'} />
            {(image.caption || image.source_url) && (
              <div className="lightbox-meta">
                {image.caption}
                {image.source_url && (
                  <a href={image.source_url} target="_blank" rel="noreferrer">
                    <ExternalLink size={12} style={{ display: 'inline', verticalAlign: -1, marginRight: 4 }} />
                    source page
                  </a>
                )}
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
