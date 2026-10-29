import { useState, useRef, useCallback } from 'react'
import { Upload, Play, Zap, Eye, Target, TrendingUp, AlertCircle, Loader2 } from 'lucide-react'
import type { AnalysisResult } from '../types'
import { uploadVideo, pollUntilDone, runDemo } from '../utils/api'

interface Props { onResult: (r: AnalysisResult) => void }

const FEATURES = [
  { icon: Eye,       label: 'Pose Estimation',   desc: 'MediaPipe 33-point skeleton overlay' },
  { icon: Target,    label: 'Ball Tracking',      desc: 'YOLOv8 trajectory arc detection' },
  { icon: Zap,       label: 'Form Analysis',      desc: 'XGBoost biomechanical scoring' },
  { icon: TrendingUp,label: 'Shot Prediction',    desc: 'Make probability with confidence' },
]

export default function UploadPage({ onResult }: Props) {
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'error'>('idle')
  const [progress, setProgress] = useState('')
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(async (f: File) => {
    setFile(f); setError(''); setStatus('uploading')
    setProgress('Uploading video...')
    try {
      const upload = await uploadVideo(f)
      setStatus('processing')
      setProgress('Extracting frames...')
      const steps = ['Running pose estimation...', 'Detecting basketball & hoop...', 'Engineering features...', 'Running ML inference...', 'Generating feedback...']
      let step = 0
      const stepTimer = setInterval(() => {
        if (step < steps.length) { setProgress(steps[step++]) }
      }, 3500)
      const stop = pollUntilDone(upload.session_id, (r) => {
        if (r.status === 'completed') {
          clearInterval(stepTimer); stop(); onResult(r)
        } else if (r.status === 'failed') {
          clearInterval(stepTimer); stop()
          setError(r.error || 'Analysis failed'); setStatus('error')
        }
      }, (e) => {
        clearInterval(stepTimer)
        setError(e.message); setStatus('error')
      })
    } catch (e: any) {
      setError(e.message || 'Analysis failed'); setStatus('error')
    }
  }, [onResult])

  const handleDemo = async () => {
    setStatus('processing'); setProgress('Running demo analysis...')
    try {
      const r = await runDemo()
      onResult(r)
    } catch (e: any) {
      setError(e.message || 'Demo failed'); setStatus('error')
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) processFile(f)
  }

  const isLoading = status === 'uploading' || status === 'processing'

  return (
    <div className="min-h-screen bg-[#0d0d1a] flex flex-col">
      {/* Header */}
      <header className="border-b border-white/[0.06] px-8 py-5 flex items-center gap-4">
        <div className="w-9 h-9 rounded-xl bg-orange-500/20 flex items-center justify-center">
          <span className="text-xl">🏀</span>
        </div>
        <div>
          <h1 className="text-white font-bold text-lg tracking-tight leading-none">HoopSense AI</h1>
          <p className="text-slate-500 text-xs mt-0.5">Basketball Shooting Analysis Platform</p>
        </div>
        <div className="ml-auto">
          <span className="text-xs bg-orange-500/10 text-orange-400 border border-orange-500/20 px-3 py-1 rounded-full font-medium">
            v1.0.0 — Computer Vision + ML
          </span>
        </div>
      </header>

      {/* Hero */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16 max-w-5xl mx-auto w-full">
        {/* Ambient background rings */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full border border-orange-500/[0.04]" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] rounded-full border border-orange-500/[0.03]" />
        </div>

        <div className="text-center mb-12 relative z-10">
          <div className="inline-flex items-center gap-2 bg-white/[0.04] border border-white/[0.08] rounded-full px-4 py-2 text-sm text-slate-400 mb-6">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            OpenCV · MediaPipe · YOLOv8 · XGBoost
          </div>
          <h2 className="text-5xl font-black text-white tracking-tight leading-tight mb-4">
            Analyse your shot.<br />
            <span className="text-orange-400">Elevate your game.</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
            Upload a shooting video and get AI-powered form scoring, biomechanical feedback, and shot make probability — powered by computer vision and machine learning.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 w-full mb-10 relative z-10">
          {FEATURES.map(({ icon: Icon, label, desc }) => (
            <div key={label} className="glass-card p-4 text-center">
              <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center mx-auto mb-3">
                <Icon size={18} className="text-orange-400" />
              </div>
              <p className="text-white text-sm font-semibold mb-1">{label}</p>
              <p className="text-slate-500 text-xs leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>

        {/* Upload zone */}
        <div className="w-full max-w-2xl relative z-10">
          {!isLoading ? (
            <>
              <div
                className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200
                  ${dragOver ? 'border-orange-400 bg-orange-500/10 scale-[1.01]' : 'border-white/[0.12] hover:border-white/25 hover:bg-white/[0.02]'}
                  ${status === 'error' ? 'border-red-500/50' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept=".mp4,.mov,.avi,.mkv,.webm"
                  className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f) }}
                />
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 transition-all
                  ${dragOver ? 'bg-orange-400 scale-110' : 'bg-white/[0.06]'}`}>
                  <Upload size={28} className={dragOver ? 'text-white' : 'text-slate-400'} />
                </div>
                {file ? (
                  <div>
                    <p className="text-white font-semibold text-lg">{file.name}</p>
                    <p className="text-slate-400 text-sm mt-1">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-white font-semibold text-lg mb-2">Drop your shooting video here</p>
                    <p className="text-slate-400 text-sm">MP4, MOV, AVI, MKV, WEBM · up to 500MB</p>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-4 flex items-start gap-3 bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                  <AlertCircle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
                  <p className="text-red-300 text-sm">{error}</p>
                </div>
              )}

              <div className="mt-4 flex items-center gap-3">
                <div className="flex-1 h-px bg-white/[0.06]" />
                <span className="text-slate-600 text-xs font-medium">or</span>
                <div className="flex-1 h-px bg-white/[0.06]" />
              </div>

              <button
                onClick={handleDemo}
                className="mt-4 w-full flex items-center justify-center gap-2 py-3.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.08] border border-white/[0.1] text-slate-300 hover:text-white transition-all text-sm font-medium group"
              >
                <Play size={16} className="text-orange-400 group-hover:scale-110 transition-transform" />
                Run demo analysis (no upload needed)
              </button>
            </>
          ) : (
            <div className="glass-card p-12 text-center">
              <div className="relative w-20 h-20 mx-auto mb-6">
                <div className="w-20 h-20 rounded-full border-2 border-orange-500/20" />
                <div className="absolute inset-0 w-20 h-20 rounded-full border-2 border-transparent border-t-orange-400 animate-spin" />
                <div className="absolute inset-4 flex items-center justify-center">
                  <span className="text-2xl">🏀</span>
                </div>
              </div>
              <p className="text-white font-semibold text-lg mb-2">{progress}</p>
              <p className="text-slate-500 text-sm">This typically takes 15–60 seconds depending on video length</p>
              <div className="mt-6 flex gap-2 justify-center">
                {['Frames', 'Pose', 'Ball', 'Features', 'ML'].map((step, i) => (
                  <div key={step} className="flex items-center gap-2">
                    <div className="flex flex-col items-center gap-1">
                      <div className={`w-2 h-2 rounded-full transition-all duration-500 ${
                        i < (['uploading','processing'].indexOf(status) + 1) * 1.5 ? 'bg-orange-400' : 'bg-white/10'
                      }`} />
                      <span className="text-[10px] text-slate-600">{step}</span>
                    </div>
                    {i < 4 && <div className="w-6 h-px bg-white/10 mb-4" />}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
