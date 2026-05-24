// Empty-state hero, restrained Aker-style. No clickable cards — pure
// visual: a serif headline, a slowly-rotating italic accent word, and a
// numbered "01/02/03" rotator below that cycles through example prompts.
//
// Aesthetic notes (matching akercompanies.com):
//   - Generous whitespace, no decorative cards or borders.
//   - Serif headline + sans-serif body for hierarchy.
//   - Numbered sections (01, 02, 03…) for rhythm.
//   - Calm motion only — slide+fade transitions, ~2.5s cadence.
//   - Single warm accent on a single italic word.
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// Italic words cycled inside the tagline.
const ROTATING_WORDS = [
  'rent rolls',
  'occupancy',
  'leases',
  'amenities',
  'floor plans',
  'photos',
  'trends',
  'move-outs',
  'charges',
]

// Generic example prompts — every one works on any property in the dataset.
// No unit numbers, no property-specific references.
const PROMPTS = [
  'What is the average rent and current occupancy?',
  'Show me the gallery and amenities.',
  'How has the rent changed over the year?',
  'Which leases are expiring in the next 90 days?',
  'List the units with the highest outstanding balance.',
  'Give me the unit-mix breakdown.',
  'What does the floor-plan page look like?',
  'How many units are vacant right now?',
]

const WORD_INTERVAL_MS = 2200
const PROMPT_INTERVAL_MS = 3400

export default function EmptyState({ propertyName }) {
  const [wordIdx, setWordIdx] = useState(0)
  const [promptIdx, setPromptIdx] = useState(0)

  useEffect(() => {
    const w = setInterval(() => setWordIdx((i) => (i + 1) % ROTATING_WORDS.length), WORD_INTERVAL_MS)
    const p = setInterval(() => setPromptIdx((i) => (i + 1) % PROMPTS.length), PROMPT_INTERVAL_MS)
    return () => { clearInterval(w); clearInterval(p) }
  }, [])

  if (!propertyName) {
    return (
      <div className="empty-aker">
        <motion.div
          className="empty-aker-overline"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          Property AI
        </motion.div>
        <motion.h1
          className="empty-aker-headline"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05, duration: 0.45 }}
        >
          Pick a property to begin.
        </motion.h1>
      </div>
    )
  }

  // Two-digit counter that follows the cycling prompt index.
  const seq = String(promptIdx + 1).padStart(2, '0')

  return (
    <div className="empty-aker">
      <motion.div
        className="empty-aker-overline"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        Property AI · {propertyName}
      </motion.div>

      <motion.h1
        className="empty-aker-headline"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05, duration: 0.5, ease: 'easeOut' }}
      >
        Ask about{' '}
        <span className="rotating-word-slot">
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={ROTATING_WORDS[wordIdx]}
              className="rotating-word"
              initial={{ y: 22, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -22, opacity: 0 }}
              transition={{ duration: 0.42, ease: [0.2, 0.65, 0.3, 0.95] }}
            >
              {ROTATING_WORDS[wordIdx]}
            </motion.span>
          </AnimatePresence>
        </span>
        .
      </motion.h1>

      <motion.div
        className="empty-aker-sub"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.25, duration: 0.5 }}
      >
        Numbers from the rent roll, photos from the marketing site, charts on
        request — quiet, scoped to this property.
      </motion.div>

      <div className="empty-aker-rotator">
        <motion.span
          key={seq}
          className="seq"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35 }}
        >
          {seq}
        </motion.span>
        <span className="rule" aria-hidden="true" />
        <div className="rotator-line">
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={PROMPTS[promptIdx]}
              className="rotator-text"
              initial={{ opacity: 0, x: 14 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -14 }}
              transition={{ duration: 0.45, ease: [0.2, 0.65, 0.3, 0.95] }}
            >
              {PROMPTS[promptIdx]}
            </motion.span>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
