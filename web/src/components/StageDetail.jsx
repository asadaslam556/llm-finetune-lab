// Metrics for whichever stage is selected (or currently running).
// Values render in mono because they're readouts, not prose.

// Nested values (the VRAM estimate) read as "key: value" lines, not "[object Object]".
function show(v) {
  if (Array.isArray(v)) return v.join(', ')
  if (v && typeof v === 'object') return Object.entries(v).map(([k, x]) => `${k}: ${x}`).join('\n')
  return String(v)
}

export default function StageDetail({ status, stageName }) {
  const stage = stageName ? status?.stages?.[stageName] : null

  return (
    <section className="panel" aria-label="Stage detail">
      <div className="panel-label">
        {stage ? `Stage detail: ${stage.label}` : 'Stage detail'}
      </div>
      {!stage && <p className="empty">Pick a stage on the rail to see its numbers.</p>}
      {stage && (
        <div className="detail-swap" key={stageName}>
          {stage.message && <p className="stage-message">{stage.message}</p>}
          {Object.keys(stage.metrics || {}).length === 0 ? (
            <p className="empty">No metrics yet, this stage hasn't finished.</p>
          ) : (
            <div className="metric-grid">
              {Object.entries(stage.metrics).map(([k, v]) => {
                const text = show(v)
                // Sentences and paths get a full row instead of a tall narrow tile.
                return (
                  <div className={text.length > 32 ? 'metric wide' : 'metric'} key={k}>
                    <div className="k">{k}</div>
                    <div className="v">{text}</div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
