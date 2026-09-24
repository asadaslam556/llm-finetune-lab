// Shows what /api/health found. Worth having on screen because most first-run
// problems (no Ollama, model not pulled, training extras missing) are exactly
// what the health probes report, and reading them here beats curling.

import { useEffect, useState } from 'react'
import { getHealth } from '../lib/api.js'

const HEALTH_POLL_MS = 15000

export default function SystemHealth() {
  const [health, setHealth] = useState(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const h = await getHealth()
      if (alive) setHealth(h)
    }
    load()
    const id = setInterval(load, HEALTH_POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [])

  if (!health) return null
  const checks = health.checks || []
  const failing = checks.filter((c) => !c.ok)

  return (
    <section className="panel" aria-label="System health">
      <div className="panel-label">
        System
        <span className="health-chip" data-state={health.status}>{health.status}</span>
        <button className="link-btn" onClick={() => setOpen((v) => !v)}>
          {open ? 'hide' : `${failing.length} of ${checks.length} failing`}
        </button>
      </div>
      {open && (
        <ul className="check-list">
          {checks.map((c) => (
            <li key={c.name} data-ok={c.ok}>
              <span className="check-name">{c.name}</span>
              <span className="check-detail">{c.detail}</span>
              <span className="check-ms">{c.ms}ms</span>
            </li>
          ))}
        </ul>
      )}
      {!open && failing.length > 0 && (
        <p className="stage-message">{failing[0].detail}</p>
      )}
      {checks.length === 0 && (health.failing || []).length > 0 && (
        <p className="stage-message">{health.failing[0]}</p>
      )}
    </section>
  )
}
