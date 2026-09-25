// Chat console. The point of this panel is comparison: ask your fine-tune
// and a hosted model the same Nimbus question and read the answers side by
// side. Each provider keeps its own history, so one model never sees (and
// apologises for) another model's reply.

import { useEffect, useRef, useState } from 'react'
import { getProviders, sendChat } from '../lib/api.js'

const SUGGESTIONS = [
  'How do I pin a note?',
  'How long do deleted notes stay in the trash?',
  'How much does Nimbus Pro cost?',
]

// Just enough formatting for chat replies: line breaks, **bold**, and
// "# heading" lines. React escapes everything, so nothing here is raw HTML.
function RichText({ text }) {
  return text.split('\n').map((line, i) => {
    const heading = /^#{1,6}\s+/.test(line)
    const body = line.replace(/^#{1,6}\s+/, '')
    const parts = body.split(/\*\*(.+?)\*\*/g).map((part, j) =>
      j % 2 ? <strong key={j}>{part}</strong> : part,
    )
    return (
      <span key={i} className={heading ? 'rt-line rt-heading' : 'rt-line'}>
        {parts}
      </span>
    )
  })
}

function Typing() {
  return (
    <span className="typing" role="status" aria-label="Waiting for the model">
      <i /><i /><i />
    </span>
  )
}

// Messages for one provider: only the turns it took part in, and only its own
// successful replies. The current (unanswered) question goes last.
function historyFor(provider, turns) {
  const out = []
  for (const t of turns) {
    if (!t.targets.includes(provider)) continue
    out.push({ role: 'user', content: t.text })
    const r = t.replies[provider]
    if (r && !r.pending && !r.error) out.push({ role: 'assistant', content: r.text })
  }
  return out
}

export default function ChatPanel() {
  const [providers, setProviders] = useState([])
  const [provider, setProvider] = useState('ollama')
  const [compareWith, setCompareWith] = useState('')
  const [input, setInput] = useState('')
  const [turns, setTurns] = useState([])
  const log = useRef(null)

  const busy = turns.some((t) => Object.values(t.replies).some((r) => r.pending))

  useEffect(() => {
    getProviders()
      .then((list) => {
        setProviders(list)
        // Start on whatever LFL_DEFAULT_PROVIDER says, not a hardcoded guess.
        const preferred = list.find((p) => p.default)
        if (preferred) setProvider(preferred.name)
      })
      .catch(() => {})
  }, [])

  // Scroll the log itself, never the page: scrollIntoView would also drag
  // the whole page down on load, hiding the header on a phone.
  useEffect(() => {
    const el = log.current
    if (el && turns.length) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [turns])

  const setReply = (id, name, reply) =>
    setTurns((all) =>
      all.map((t) => (t.id === id ? { ...t, replies: { ...t.replies, [name]: reply } } : t)),
    )

  const ask = async (raw) => {
    const text = raw.trim()
    if (!text || busy) return
    setInput('')
    const targets = [...new Set([provider, compareWith].filter(Boolean))]
    const turn = {
      id: Date.now(),
      text,
      targets,
      replies: Object.fromEntries(targets.map((n) => [n, { pending: true }])),
    }
    const next = [...turns, turn]
    setTurns(next)

    await Promise.all(
      targets.map(async (name) => {
        try {
          const res = await sendChat(name, historyFor(name, next))
          setReply(turn.id, name, {
            text: res.reply,
            model: res.model,
            ms: res.latency_ms,
            error: !res.ok,
          })
        } catch (e) {
          setReply(turn.id, name, { text: e.message, error: true })
        }
      }),
    )
  }

  const options = providers.length ? providers : [{ name: 'ollama', configured: true }]
  const label = (p) => `${p.name}${p.configured ? '' : ' (not configured)'}`

  return (
    <section className="panel" aria-label="Chat console">
      <div className="panel-label">Chat console</div>

      <div className="chat-log" aria-live="polite" ref={log}>
        {turns.length === 0 && (
          <div className="chat-empty">
            <p className="empty">
              Ask a Nimbus question. Turn on <strong>Compare</strong> to put your fine-tune
              next to a hosted model.
            </p>
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="chip" onClick={() => ask(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t) => (
          <div className="turn" key={t.id}>
            <div className="msg user">
              <div className="who">you</div>
              {t.text}
            </div>
            <div className={`replies${t.targets.length > 1 ? ' replies-compare' : ''}`}>
              {t.targets.map((name) => {
                const r = t.replies[name]
                return (
                  <div key={name} className={`msg assistant${r.error ? ' error' : ''}`}>
                    <div className="who">
                      {name}
                      {r.model && <> · {r.model}</>}
                      {r.ms > 0 && <span className="ms">{(r.ms / 1000).toFixed(1)}s</span>}
                    </div>
                    {r.pending ? <Typing /> : <RichText text={r.text} />}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      <form
        className="chat-form"
        onSubmit={(e) => {
          e.preventDefault()
          ask(input)
        }}
      >
        <select
          value={provider}
          onChange={(e) => {
            setProvider(e.target.value)
            if (e.target.value === compareWith) setCompareWith('')
          }}
          aria-label="Provider"
        >
          {options.map((p) => (
            <option key={p.name} value={p.name}>{label(p)}</option>
          ))}
        </select>
        <select
          value={compareWith}
          onChange={(e) => setCompareWith(e.target.value)}
          aria-label="Compare with"
          className={compareWith ? 'compare-on' : ''}
        >
          <option value="">Compare: off</option>
          {options
            .filter((p) => p.name !== provider)
            .map((p) => (
              <option key={p.name} value={p.name}>vs {label(p)}</option>
            ))}
        </select>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message the model"
          aria-label="Message"
        />
        <button className="btn btn-primary" type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </section>
  )
}
