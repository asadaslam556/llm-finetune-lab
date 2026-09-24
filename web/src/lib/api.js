// Thin fetch wrappers. Errors come back as thrown Error(message) carrying the
// server's own words, so components never have to invent an explanation.

function readableError(body, status) {
  const d = body?.detail
  if (typeof d === 'string') return d
  // FastAPI validation errors arrive as an array of objects. Passing that
  // straight to new Error() renders "[object Object]", which tells the user
  // precisely nothing, so flatten it into field: message pairs.
  if (Array.isArray(d)) {
    return d
      .map((item) => {
        const where = Array.isArray(item.loc) ? item.loc.filter((p) => p !== 'body').join('.') : ''
        return where ? `${where}: ${item.msg}` : item.msg
      })
      .join('; ')
  }
  if (d && typeof d === 'object') return JSON.stringify(d)
  return `HTTP ${status}`
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error('Cannot reach the backend. Is uvicorn running on port 8000?')
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(readableError(body, res.status))
  return body
}

export const getStatus = () => request('/api/pipeline/status')
export const getProviders = () => request('/api/providers')
// /api/health answers 503 when a required check fails, and that body is
// exactly the one worth showing, so it cannot go through request(), which
// throws away the body of any non-2xx response.
export async function getHealth() {
  try {
    const res = await fetch('/api/health')
    const body = await res.json().catch(() => null)
    if (body && Array.isArray(body.checks)) return body
    return { status: 'error', checks: [], failing: [`HTTP ${res.status}`] }
  } catch {
    return { status: 'error', checks: [], failing: ['Cannot reach the backend. Is uvicorn running on port 8000?'] }
  }
}
export const startRun = (dryRun, stages = null) =>
  request('/api/pipeline/run', {
    method: 'POST',
    body: JSON.stringify({ dry_run: dryRun, stages }),
  })
export const sendChat = (provider, messages) =>
  request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ provider, messages }),
  })
