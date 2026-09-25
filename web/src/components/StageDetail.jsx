// Metrics for whichever stage is selected (or currently running).
// Values render in mono because they're readouts, not prose.

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
              {Object.entries(stage.metrics).map(([k, v]) => (
                <div className="metric" key={k}>
                  <div className="k">{k}</div>
                  <div className="v">{String(v)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
