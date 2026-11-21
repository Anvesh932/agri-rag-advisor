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

export async function health() {
  const res = await fetch(`${API}/health`)
  return res.json()
}
