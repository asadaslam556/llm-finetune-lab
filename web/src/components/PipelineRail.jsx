// The signature element: seven nodes on one conduit. Segment colours track
// stage state, the active node pulses and ticks its elapsed seconds.

import { useEffect, useState } from 'react'

const ORDER = ['ingest', 'prepare', 'pull_base', 'profile', 'finetune', 'evaluate', 'export_deploy']

// Counts from the stage's real server-side start time, not from whenever this
// component happened to mount. Otherwise a page reload mid-run restarts the
// clock at zero and quietly lies about how long training has been going.
function ElapsedTicker({ startedAt }) {
  const since = () => (startedAt ? Math.max(0, Math.round(Date.now() / 1000 - startedAt)) : 0)
  const [secs, setSecs] = useState(since)

  useEffect(() => {
    setSecs(since())
    const id = setInterval(() => setSecs(since()), 1000)
    return () => clearInterval(id)
  }, [startedAt])

  return <>{secs}s</>
}

// Drawn marks rather than unicode glyphs, so they render the same on every
// font and can animate their stroke in.
function glyph(state, index) {
  if (state === 'done') {
    return (
      <svg viewBox="0 0 12 12" className="mark">
        <path d="M2.5 6.4 5 8.8 9.5 3.6" pathLength="1" />
      </svg>
    )
  }
  if (state === 'failed') {
    return (
      <svg viewBox="0 0 12 12" className="mark">
        <path d="M3.2 3.2 8.8 8.8M8.8 3.2 3.2 8.8" pathLength="1" />
      </svg>
    )
  }
  return String(index + 1)
}

function meta(stage, idle) {
  switch (stage.state) {
    case 'running': return <>running · <ElapsedTicker startedAt={stage.started_at} /></>
    case 'done': return `${stage.seconds}s`
    case 'failed': return 'failed'
    case 'skipped': return 'skipped'
    default: return idle ? 'waiting' : 'queued'
  }
}

export default function PipelineRail({ status, selected, onSelect }) {
  if (!status) return <p className="empty">Connecting to backend…</p>

  const idle = status.state === 'idle'

  return (
    <>
      <ol className="rail">
        {ORDER.map((name, i) => {
          const stage = status.stages[name]
          return (
            <li key={name} className="rail-stage" data-state={stage.state} data-selected={selected === name}>
              <button
                type="button"
                className="rail-hit"
                onClick={() => onSelect(name)}
                aria-current={status.current_stage === name ? 'step' : undefined}
              >
                <span className="node" aria-hidden="true">{glyph(stage.state, i)}</span>
                <span className="stage-text">
                  <span className="stage-name">{stage.label}</span>
                  <span className="stage-meta">{meta(stage, idle)}</span>
                </span>
              </button>
            </li>
          )
        })}
      </ol>
      {idle && (
        <p className="empty" style={{ marginTop: 14 }}>
          No run yet. Start a dry run to watch all seven stages.
        </p>
      )}
    </>
  )
}
