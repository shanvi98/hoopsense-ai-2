import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import DashboardPage from './pages/DashboardPage'
import type { AnalysisResult } from './types'

export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null)

  if (result?.status === 'completed' && result.result) {
    return <DashboardPage result={result} onReset={() => setResult(null)} />
  }

  return <UploadPage onResult={setResult} />
}
