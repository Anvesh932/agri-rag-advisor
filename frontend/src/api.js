const API = '/api'

export async function askQuestion({ question, lang, model, region }) {
  const res = await fetch(`${API}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, lang, model, region: region || null }),
  })
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function transcribe(blob) {
  const form = new FormData()
  form.append('file', blob, 'recording.webm')
  const res = await fetch(`${API}/transcribe`, { method: 'POST', body: form })
  if (!res.ok) {
    let msg = `transcription error ${res.status}`
    try { msg = (await res.json()).detail || msg } catch {}
    throw new Error(msg)
  }
  return res.json()
}

export async function health() {
  const res = await fetch(`${API}/health`)
  return res.json()
}
