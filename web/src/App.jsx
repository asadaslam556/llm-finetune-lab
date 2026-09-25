import { useCallback, useEffect, useRef, useState } from 'react'
import { getStatus } from './lib/api.js'
import PipelineRail from './components/PipelineRail.jsx'
import RunControls from './components/RunControls.jsx'
import StageDetail from './components/StageDetail.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import SystemHealth from './components/SystemHealth.jsx'

const POLL_MS = 1500

export default function App() {
  const [status, setStatus] = useState(null)
  const [selected, setSelected] = useState(null)
  const [offline, setOffline] = useState(false)
  const alive = useRef(true)

  const tick = useCallback(async () => {
    try {
      const s = await getStatus()
      if (!alive.current) return
      setStatus(s)
      setOffline(false)
    } catch {
      if (alive.current) setOffline(true)
    }
  }, [])

  useEffect(() => {
    alive.current = true
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => {
      alive.current = false
      clearInterval(id)
    }
  }, [tick])

  const state = offline ? 'offline' : (status?.state ?? 'idle')
  // The rail follows the running stage on its own; clicking a stage pins the
  // detail panel to that one until you pick another.
  const shownStage = selected ?? status?.current_stage ?? null

  // Share of the requested stages that have finished, for the header line.
  const stages = Object.values(status?.stages ?? {}).filter((s) => s.state !== 'skipped')
  const finished = stages.filter((s) => s.state === 'done').length
  const progress = stages.length ? finished / stages.length : 0

  return (
    <>
      <header className="console-header">
        <span className="dot" aria-hidden="true" />
        <h1>llm-finetune-lab</h1>
        <span className="sub">Nimbus support model · QLoRA pipeline</span>
        <span className="state-chip" data-state={state}>
          {offline ? 'backend offline' : state}
        </span>
        <span
          className="run-progress"
          data-state={state}
          style={{ '--p': progress }}
          role="progressbar"
          aria-label="Pipeline progress"
          aria-valuemin={0}
          aria-valuemax={stages.length || 7}
          aria-valuenow={finished}
        />
      </header>

      <main className="console-body">
        <section className="panel" aria-label="Pipeline stages">
          <div className="panel-label">Pipeline</div>
          <PipelineRail status={status} selected={shownStage} onSelect={setSelected} />
        </section>

        <div>
          <RunControls status={status} offline={offline} onStarted={tick} />
          <StageDetail status={status} stageName={shownStage} />
          <ChatPanel />
          <SystemHealth />
        </div>
      </main>

      <footer className="console-footer">
        <span>llm-finetune-lab</span>
        <span>built by Asad Aslam</span>
      </footer>
    </>
  )
}
