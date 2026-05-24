// Animated empty-state hero for the chat surface. Shows a rotating tagline
// over a grid of suggestion cards (each with an icon and example prompt).
// Click any card to submit that prompt immediately.
import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  TrendingUp, ListChecks, Wallet, CalendarClock,
  Image as ImageIcon, BarChart3, Building, Sparkles,
} from 'lucide-react'

// Six suggestion cards covering the agent's main capability surfaces.
// The ordering balances financial / leasing / operations / visual / chart so
// no two adjacent cards feel like duplicates.
const CARDS = [
  {
    icon: TrendingUp,
    title: 'Rent trend',
    prompt: 'How has the average rent changed over the year?',
    hint: 'Chart',
  },
  {
    icon: ListChecks,
    title: 'Unit mix',
    prompt: 'Show me the unit mix breakdown.',
    hint: 'Table',
  },
  {
    icon: Wallet,
    title: 'Top balances',
    prompt: 'Which units have the highest outstanding balance?',
    hint: 'List',
  },
  {
    icon: CalendarClock,
    title: 'Expiring leases',
    prompt: 'Which leases are expiring in the next 90 days?',
    hint: 'List',
  },
  {
    icon: ImageIcon,
    title: 'Amenities & gallery',
    prompt: 'Show me the gallery and amenities',
    hint: 'Photos',
  },
  {
    icon: BarChart3,
    title: 'Compare units',
    prompt: 'Compare units 301 and 302 — rent, sqft, market rent.',
    hint: 'Chart',
  },
]

// Rotating one-word descriptors that loop under the headline. Pure visual
// flourish to signal "this thing does many things".
const ROTATING_WORDS = [
  'rent',
  'leases',
  'occupancy',
  'amenities',
  'floor plans',
  'charges',
  'photos',
  'trends',
]

const fadeUp = {
  hidden: { opacity: 0, y: 8 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.04 * i, duration: 0.32, ease: 'easeOut' },
  }),
}

export default function EmptyState({ propertyName, onPick, disabled }) {
  const [wordIndex, setWordIndex] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setWordIndex((i) => (i + 1) % ROTATING_WORDS.length)
    }, 2200)
    return () => clearInterval(id)
  }, [])

  const greeting = useMemo(() => {
    if (!propertyName) return 'Pick a property to start'
    return propertyName
  }, [propertyName])

  return (
    <div className="empty-hero">
      <motion.div
        className="empty-hero-head"
        variants={fadeUp}
        initial="hidden"
        animate="visible"
      >
        <div className="empty-hero-eyebrow">
          <Sparkles size={12} />
          <span>Property AI</span>
        </div>
        <h2>
          <span className="empty-hero-greet">
            <Building size={22} aria-hidden="true" />
            {greeting}
          </span>
        </h2>
        {propertyName && (
          <p className="empty-hero-sub">
            Ask about{' '}
            <span className="rotating-word-slot">
              <AnimatePresence mode="wait" initial={false}>
                <motion.span
                  key={ROTATING_WORDS[wordIndex]}
                  className="rotating-word"
                  initial={{ y: 14, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ y: -14, opacity: 0 }}
                  transition={{ duration: 0.36, ease: 'easeOut' }}
                >
                  {ROTATING_WORDS[wordIndex]}
                </motion.span>
              </AnimatePresence>
            </span>
            {' '}— or pick a card below.
          </p>
        )}
      </motion.div>

      {propertyName && (
        <motion.div
          className="empty-hero-grid"
          initial="hidden"
          animate="visible"
        >
          {CARDS.map((c, i) => {
            const Icon = c.icon
            return (
              <motion.button
                key={c.title}
                type="button"
                className="empty-card"
                disabled={disabled}
                custom={i + 1}
                variants={fadeUp}
                whileHover={disabled ? undefined : { y: -2 }}
                whileTap={disabled ? undefined : { scale: 0.98 }}
                onClick={() => !disabled && onPick(c.prompt)}
              >
                <span className="empty-card-icon">
                  <Icon size={16} />
                </span>
                <span className="empty-card-body">
                  <span className="empty-card-title">{c.title}</span>
                  <span className="empty-card-prompt">{c.prompt}</span>
                </span>
                <span className="empty-card-hint">{c.hint}</span>
              </motion.button>
            )
          })}
        </motion.div>
      )}
    </div>
  )
}
