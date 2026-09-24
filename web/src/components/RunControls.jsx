// Start buttons. Dry run is the primary path: it exercises every stage with
// zero downloads. Real run is there for machines with the training extras
// installed; the backend explains itself if they're missing.

import { useState } from 'react'
import { startRun } from '../lib/api.js'

export default function RunControls({ status, offline, onStarted }) {
  const [error, setError] = useState(null)
  const [pending, setPending] = useState(false)

  // `busy` is the backend's live lock state, which is the honest answer.
  // status.state alone lags it by up to a poll, and status===null on first
  // paint would otherwise leave the buttons enabled before we know anything.
  const busy = status?.busy || status?.state === 'running'
  const disabled = busy || pending || offline || status === null

  const launch = async (dry) => {
    setError(null)
    setPending(true)
    try {
      await startRun(dry)
      await onStarted?.()  // repaint immediately instead of waiting for the next poll
    } catch (e) {
      setError(e.message)
    } finally {
      setPending(false)
    }
  }

  const note = () => {
    if (offline) return 'Backend unreachable. Start uvicorn on port 8000.'
    if (busy) return 'Run in progress, one at a time.'
    return 'Dry run: a safe rehearsal, no GPU needed. Real run: needs an NVIDIA GPU and the training libraries (no GPU? use the Colab notebook).'
  }

  return (
    <section className="panel" aria-label="Run controls">
      <div className="panel-label">Run</div>
      <div className="controls-row">
        <button className="btn btn-primary" disabled={disabled} onClick={() => launch(true)}>
          Start dry run
        </button>
        <button className="btn" disabled={disabled} onClick={() => launch(false)}>
          Start real run
        </button>
        <span className="controls-note">{note()}</span>
      </div>
      {error && <div className="controls-error" role="alert">{error}</div>}
    </section>
  )
}
