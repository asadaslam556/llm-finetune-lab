// Chat console. Provider picker is fed by /api/providers; unconfigured
// providers stay selectable on purpose. Picking one shows the backend's
// own explanation of what's missing, which is more instructive than
// hiding the option.

import { useEffect, useRef, useState } from 'react'
import { getProviders, sendChat } from '../lib/api.js'

export default function ChatPanel() {
  const [providers, setProviders] = useState([])
  const [provider, setProvider] = useState('ollama')
  const [input, setInput] = useState('')
  const [log, setLog] = useState([])
  const [busy, setBusy] = useState(false)
  const bottom = useRef(null)

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

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log])

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    const nextLog = [...log, { who: 'you', text }]
    setLog(nextLog)
    setBusy(true)
    try {
      const history = nextLog
        .filter((m) => !m.error)
        .map((m) => ({ role: m.who === 'you' ? 'user' : 'assistant', content: m.text }))
      const res = await sendChat(provider, history)
      setLog((l) => [...l, {
        who: `${res.provider} · ${res.model}`,
        text: res.reply,
        error: !res.ok,
      }])
    } catch (e) {
      setLog((l) => [...l, { who: 'error', text: e.message, error: true }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="panel" aria-label="Chat console">
      <div className="panel-label">Chat console</div>
      <div className="chat-log">
        {log.length === 0 && (
          <p className="empty">
            Ask the model something Nimbus-flavored. “How do I pin a note?” is the classic.
          </p>
        )}
        {log.map((m, i) => (
          <div key={i} className={`msg ${m.who === 'you' ? 'user' : 'assistant'} ${m.error ? 'error' : ''}`}>
            <div className="who">{m.who}</div>
            {m.text}
          </div>
        ))}
        <div ref={bottom} />
      </div>
      <div className="chat-form">
        <select
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          aria-label="Provider"
        >
          {providers.length === 0 && <option value="ollama">ollama</option>}
          {providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}{p.configured ? '' : ' (not configured)'}
            </option>
          ))}
        </select>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Message the model"
          aria-label="Message"
        />
        <button className="btn btn-primary" onClick={send} disabled={busy}>
          Send
        </button>
      </div>
    </section>
  )
}
