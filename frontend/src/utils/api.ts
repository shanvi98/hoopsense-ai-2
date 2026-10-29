import type { AnalysisResult } from '../types'

const BASE = '/api/v1'

export async function uploadVideo(file: File): Promise<{ session_id: string; status: string; message: string }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/analysis/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
  return res.json()
}

export async function pollStatus(sessionId: string): Promise<AnalysisResult> {
  const res = await fetch(`${BASE}/analysis/status/${sessionId}`)
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

export async function runDemo(): Promise<AnalysisResult> {
  const res = await fetch(`${BASE}/analysis/demo`, { method: 'POST' })
  if (!res.ok) throw new Error('Demo failed')
  return res.json()
}

export function pollUntilDone(
  sessionId: string,
  onUpdate: (r: AnalysisResult) => void,
  onError: (e: Error) => void,
  intervalMs = 2000
): () => void {
  const timer = setInterval(async () => {
    try {
      const result = await pollStatus(sessionId)
      onUpdate(result)
      if (result.status === 'completed' || result.status === 'failed') clearInterval(timer)
    } catch (e) {
      clearInterval(timer)
      onError(e instanceof Error ? e : new Error('Status check failed'))
    }
  }, intervalMs)
  return () => clearInterval(timer)
}
