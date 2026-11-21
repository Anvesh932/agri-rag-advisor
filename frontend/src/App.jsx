import { useState } from 'react'
import { askQuestion } from './api.js'

const LANGS = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिन्दी (Hindi)' },
  { code: 'bn', label: 'বাংলা (Bengali)' },
  { code: 'ta', label: 'தமிழ் (Tamil)' },
]

const MODELS = [
  { key: 'llama-3.2-3b', label: 'Llama-3.2-3B (detailed)' },
  { key: 'falcon-7b', label: 'Falcon-7B (concise)' },
  { key: 'mistral-7b', label: 'Mistral-7B (most accurate)' },
  { key: 'phi-2', label: 'Phi-2 (lightweight)' },
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [lang, setLang] = useState('en')
  const [model, setModel] = useState('llama-3.2-3b')
  const [region, setRegion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function submit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null)
    try {
      setResult(await askQuestion({ question, lang, model, region }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={styles.app}>
      <h1 style={styles.h1}>🌾 Agri-RAG Advisor</h1>
      <p style={styles.sub}>Region-specific farming advice for South Asia · SDG 12</p>

      <form onSubmit={submit} style={styles.form}>
        <textarea
          value={question} onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How can I improve my soil fertility without chemical fertilizers?"
          rows={3} required style={styles.textarea}
        />
        <div style={styles.row}>
          <select value={lang} onChange={(e) => setLang(e.target.value)} style={styles.select}>
            {LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
          <select value={model} onChange={(e) => setModel(e.target.value)} style={styles.select}>
            {MODELS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
          <input value={region} onChange={(e) => setRegion(e.target.value)}
            placeholder="Region (optional, e.g. Indo-Gangetic Plain)" style={{ ...styles.select, flex: 2 }} />
        </div>
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? 'Asking…' : 'Get advice'}
        </button>
      </form>

      {error && <p style={styles.error}>⚠ {error}</p>}

      {result && (
        <section style={styles.card}>
          <h3 style={styles.cardTitle}>Advice</h3>
          <p style={styles.answer}>{result.answer}</p>
          {result.guardrails?.flags?.length > 0 && (
            <p style={styles.flag}>⚠ Guardrails: {result.guardrails.flags.join('; ')}</p>
          )}
          {result.sources?.length > 0 && (
            <details>
              <summary style={styles.summary}>Sources ({result.sources.length})</summary>
              <ul style={styles.sources}>
                {result.sources.map((s, i) => (
                  <li key={i}>
                    {s.title} {s.year ? `(${s.year})` : ''} — score {s.score}
                    {s.flag ? ` · ⚠ ${s.flag}` : ''}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
    </main>
  )
}

const styles = {
  app: { maxWidth: 640, margin: '2rem auto', padding: '0 1rem', fontFamily: 'system-ui, sans-serif' },
  h1: { marginBottom: 4 },
  sub: { color: '#555', marginTop: 0 },
  form: { display: 'flex', flexDirection: 'column', gap: 10, marginTop: 20 },
  textarea: { padding: 10, fontSize: 15, border: '1px solid #ccc', borderRadius: 8 },
  row: { display: 'flex', gap: 8 },
  select: { padding: 8, borderRadius: 8, border: '1px solid #ccc', flex: 1 },
  button: { padding: '10px 16px', fontSize: 15, background: '#2e7d32', color: '#fff', border: 0, borderRadius: 8, cursor: 'pointer' },
  error: { color: '#b00020' },
  card: { marginTop: 20, padding: 16, background: '#f4faf4', border: '1px solid #c8e6c9', borderRadius: 10 },
  cardTitle: { margin: '0 0 8px' },
  answer: { whiteSpace: 'pre-wrap', lineHeight: 1.5 },
  flag: { color: '#a15c00', fontSize: 13 },
  summary: { cursor: 'pointer', color: '#2e7d32' },
  sources: { fontSize: 13, color: '#444' },
}
