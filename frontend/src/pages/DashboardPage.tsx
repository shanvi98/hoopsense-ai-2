import { useState, useRef, useEffect } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area
} from 'recharts'
import { ArrowLeft, Download, ChevronDown, ChevronUp, CheckCircle2, AlertTriangle, XCircle, Activity, Target, Zap, TrendingUp, Clock, Ruler } from 'lucide-react'
import type { AnalysisResult, FeedbackItem, BiomechanicalFeatures } from '../types'

interface Props { result: AnalysisResult; onReset: () => void }

function gradeColor(grade: string) {
  return grade === 'Elite' ? '#22c55e' : grade === 'Good' ? '#f97316' : grade === 'Average' ? '#eab308' : '#ef4444'
}

function MakeGauge({ probability }: { probability: number }) {
  const pct = Math.round(probability * 100)
  const angle = (pct / 100) * 180 - 90
  const r = 80, cx = 100, cy = 100
  const startAngle = -180
  const endAngle = startAngle + (pct / 100) * 180
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const arcX = (a: number) => cx + r * Math.cos(toRad(a))
  const arcY = (a: number) => cy + r * Math.sin(toRad(a))
  const largeArc = pct > 50 ? 1 : 0

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 110" className="w-48">
        <path d={`M ${arcX(-180)} ${arcY(-180)} A ${r} ${r} 0 1 1 ${arcX(0)} ${arcY(0)}`}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12" strokeLinecap="round" />
        {pct > 0 && (
          <path d={`M ${arcX(-180)} ${arcY(-180)} A ${r} ${r} 0 ${largeArc} 1 ${arcX(endAngle)} ${arcY(endAngle)}`}
            fill="none" stroke={pct >= 60 ? '#22c55e' : pct >= 40 ? '#f97316' : '#ef4444'}
            strokeWidth="12" strokeLinecap="round" />
        )}
        <text x="100" y="90" textAnchor="middle" className="text-4xl font-black" fill="white"
          style={{ fontSize: 28, fontWeight: 800 }}>{pct}%</text>
        <text x="100" y="108" textAnchor="middle" fill="#64748b" style={{ fontSize: 10 }}>make probability</text>
      </svg>
      <p className="text-slate-400 text-xs mt-1">
        {pct >= 60 ? '🟢 High confidence shot' : pct >= 40 ? '🟡 Moderate chance' : '🔴 Low probability'}
      </p>
    </div>
  )
}

function ScoreRing({ score, grade }: { score: number; grade: string }) {
  const color = gradeColor(grade)
  const circumference = 2 * Math.PI * 54
  const offset = circumference * (1 - score / 100)

  return (
    <div className="relative w-36 h-36 flex items-center justify-center">
      <svg viewBox="0 0 120 120" className="w-36 h-36 -rotate-90 absolute">
        <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
        <circle cx="60" cy="60" r="54" fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round" style={{ transition: 'stroke-dashoffset 1s ease' }} />
      </svg>
      <div className="text-center z-10">
        <p className="text-3xl font-black text-white leading-none">{Math.round(score)}</p>
        <p className="text-xs font-bold mt-1" style={{ color }}>{grade}</p>
      </div>
    </div>
  )
}

function FeedbackCard({ item }: { item: FeedbackItem }) {
  const [open, setOpen] = useState(false)
  const Icon = item.rating === 'good' ? CheckCircle2 : item.rating === 'warning' ? AlertTriangle : XCircle
  const colors = {
    good: 'text-green-400 bg-green-500/10 border-green-500/20',
    warning: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    error: 'text-red-400 bg-red-500/10 border-red-500/20',
  }

  return (
    <div className={`border rounded-xl p-4 cursor-pointer transition-all ${colors[item.rating]} ${open ? 'ring-1 ring-current/20' : ''}`}
      onClick={() => setOpen(!open)}>
      <div className="flex items-center gap-3">
        <Icon size={16} className="flex-shrink-0" />
        <span className="font-semibold text-sm flex-1">{item.aspect}</span>
        {item.metric_value !== undefined && (
          <span className="text-xs opacity-70 font-mono">{item.metric_value.toFixed ? item.metric_value.toFixed(2) : item.metric_value}</span>
        )}
        {open ? <ChevronUp size={14} className="opacity-60" /> : <ChevronDown size={14} className="opacity-60" />}
      </div>
      {open && (
        <div className="mt-3 pt-3 border-t border-current/20">
          <p className="text-sm opacity-90 leading-relaxed">{item.message}</p>
          {item.benchmark_range && (
            <p className="text-xs opacity-60 mt-2">Benchmark: {item.benchmark_range}</p>
          )}
        </div>
      )}
    </div>
  )
}

function TrajectoryViz({ trajectory, hoop }: { trajectory: any[]; hoop: any }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !trajectory.length) return
    const ctx = canvas.getContext('2d')!
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    // Court background
    ctx.fillStyle = '#0d0d1a'
    ctx.fillRect(0, 0, W, H)

    // Draw hoop
    if (hoop) {
      const hx = hoop.x * W, hy = hoop.y * H
      ctx.strokeStyle = '#f97316'
      ctx.lineWidth = 3
      ctx.strokeRect(hx - hoop.w * W / 2, hy - 8, hoop.w * W, 8)
      ctx.strokeStyle = 'rgba(249,115,22,0.3)'
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx, hy + H * 0.4)
      ctx.stroke(); ctx.setLineDash([])
    }

    // Draw trajectory arc
    if (trajectory.length > 1) {
      const grad = ctx.createLinearGradient(0, H, 0, 0)
      grad.addColorStop(0, 'rgba(59,130,246,0.8)')
      grad.addColorStop(0.5, 'rgba(249,115,22,0.9)')
      grad.addColorStop(1, 'rgba(34,197,94,0.8)')

      ctx.beginPath()
      ctx.strokeStyle = grad
      ctx.lineWidth = 3
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      trajectory.forEach((pt, i) => {
        const px = pt.x * W, py = pt.y * H
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
      })
      ctx.stroke()

      // Ball dots along path
      trajectory.filter((_, i) => i % 4 === 0).forEach((pt, i, arr) => {
        const px = pt.x * W, py = pt.y * H
        const alpha = i / arr.length
        ctx.beginPath()
        ctx.arc(px, py, 4 + alpha * 3, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(249,115,22,${0.4 + alpha * 0.6})`
        ctx.fill()
      })

      // Release point marker
      const last = trajectory[trajectory.length - 1]
      ctx.beginPath()
      ctx.arc(last.x * W, last.y * H, 8, 0, Math.PI * 2)
      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = 2
      ctx.stroke()
    }
  }, [trajectory, hoop])

  return (
    <div className="relative">
      <canvas ref={canvasRef} width={400} height={250} className="w-full rounded-xl bg-[#0d0d1a]" />
      <div className="absolute bottom-3 left-3 flex gap-3 text-xs">
        <span className="flex items-center gap-1.5"><span className="w-3 h-1 bg-blue-400 rounded inline-block" />Launch</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-1 bg-orange-400 rounded inline-block" />Peak arc</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-1 bg-green-400 rounded inline-block" />Release</span>
      </div>
    </div>
  )
}

function SkeletonViz({ poseFrames }: { poseFrames: any[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frameIdx, setFrameIdx] = useState(0)

  const CONNECTIONS = [
    ['left_shoulder','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],
    ['right_shoulder','right_elbow'],['right_elbow','right_wrist'],
    ['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],
    ['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle'],
  ]

  const PHASE_COLORS: Record<string, string> = {
    gather: '#eab308', setup: '#3b82f6', release: '#f97316', follow_through: '#22c55e', unknown: '#64748b'
  }

  useEffect(() => {
    if (!poseFrames.length) return
    const timer = setInterval(() => setFrameIdx(i => (i + 1) % poseFrames.length), 150)
    return () => clearInterval(timer)
  }, [poseFrames.length])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !poseFrames.length) return
    const pf = poseFrames[frameIdx]
    if (!pf) return
    const ctx = canvas.getContext('2d')!
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    const phaseColor = PHASE_COLORS[pf.phase] || '#64748b'
    const lms = pf.landmarks

    // Draw connections
    ctx.strokeStyle = phaseColor
    ctx.lineWidth = 2.5
    ctx.lineCap = 'round'
    CONNECTIONS.forEach(([a, b]) => {
      const la = lms[a], lb = lms[b]
      if (la && lb && la.visibility > 0.3 && lb.visibility > 0.3) {
        ctx.globalAlpha = 0.7
        ctx.beginPath()
        ctx.moveTo(la.x * W, la.y * H)
        ctx.lineTo(lb.x * W, lb.y * H)
        ctx.stroke()
      }
    })

    // Draw joints
    ctx.globalAlpha = 1
    Object.values(lms).forEach((lm: any) => {
      if (lm.visibility > 0.3) {
        ctx.beginPath()
        ctx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI * 2)
        ctx.fillStyle = phaseColor
        ctx.fill()
        ctx.strokeStyle = '#0d0d1a'
        ctx.lineWidth = 1.5
        ctx.stroke()
      }
    })

    // Draw ball
    if (pf.ball) {
      ctx.beginPath()
      ctx.arc(pf.ball.x * W, pf.ball.y * H, 10, 0, Math.PI * 2)
      ctx.strokeStyle = '#f97316'
      ctx.lineWidth = 2.5
      ctx.stroke()
      ctx.fillStyle = 'rgba(249,115,22,0.2)'
      ctx.fill()
    }
  }, [frameIdx, poseFrames])

  const pf = poseFrames[frameIdx]
  const phaseColor = pf ? (PHASE_COLORS[pf.phase] || '#64748b') : '#64748b'

  return (
    <div className="relative">
      <canvas ref={canvasRef} width={300} height={400} className="w-full rounded-xl bg-[#0d0d1a]" />
      {pf && (
        <div className="absolute top-3 right-3 text-xs px-2.5 py-1 rounded-full font-semibold capitalize"
          style={{ background: `${phaseColor}22`, color: phaseColor, border: `1px solid ${phaseColor}44` }}>
          {pf.phase.replace('_', ' ')}
        </div>
      )}
      <div className="absolute bottom-3 left-3 text-xs text-slate-500">
        Frame {pf?.frame_idx ?? 0} · {Math.round((pf?.timestamp_ms ?? 0) / 100) / 10}s
      </div>
    </div>
  )
}

function RadarViz({ features }: { features: BiomechanicalFeatures }) {
  const data = [
    { axis: 'Elbow', value: Math.round(Math.min(100, (features.elbow_angle_release / 110) * 100)) },
    { axis: 'Release', value: Math.round(features.release_timing_score * 100) },
    { axis: 'Follow', value: Math.round(features.follow_through_score * 100) },
    { axis: 'Balance', value: Math.round(features.balance_score * 100) },
    { axis: 'Rhythm', value: Math.round(features.rhythm_consistency * 100) },
    { axis: 'Arc', value: Math.round(Math.max(0, 100 - Math.abs(features.ball_arc_angle - 46.5) * 2)) },
  ]

  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
        <PolarGrid stroke="rgba(255,255,255,0.08)" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: '#64748b', fontSize: 11 }} />
        <Radar name="Score" dataKey="value" stroke="#f97316" fill="#f97316" fillOpacity={0.15} strokeWidth={2} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

export default function DashboardPage({ result, onReset }: Props) {
  const pred = result.result!
  const f = pred.features
  const [activeTab, setActiveTab] = useState<'overview' | 'biomechanics' | 'feedback'>('overview')

  const biomechanicsData = [
    { name: 'Elbow angle',   value: f.elbow_angle_release, unit: '°',  good: [80, 110] },
    { name: 'Knee bend',     value: f.knee_bend_max,       unit: '°',  good: [20, 60] },
    { name: 'Wrist snap',    value: f.wrist_release_angle, unit: '°',  good: [30, 70] },
    { name: 'Shot arc',      value: f.ball_arc_angle,      unit: '°',  good: [38, 55] },
    { name: 'Arm extension', value: Math.round(f.shooting_arm_extension * 100), unit: '%', good: [80, 100] },
    { name: 'Jump height',   value: Math.round(f.jump_height_normalized * 100), unit: '%', good: [5, 30] },
    { name: 'Release timing',value: Math.round(f.release_timing_score * 100), unit: '%', good: [60, 100] },
    { name: 'Follow-through',value: Math.round(f.follow_through_score * 100), unit: '%', good: [70, 100] },
    { name: 'Balance',       value: Math.round(f.balance_score * 100), unit: '%', good: [70, 100] },
    { name: 'Rhythm',        value: Math.round(f.rhythm_consistency * 100), unit: '%', good: [60, 100] },
  ]

  const trajectoryVelocity = pred.ball_trajectory.map((pt, i) => ({
    frame: pt.frame,
    speed: Math.round(Math.sqrt(pt.vx ** 2 + pt.vy ** 2) * 1000) / 10,
  }))

  const errors   = pred.feedback.filter(fb => fb.rating === 'error').length
  const warnings = pred.feedback.filter(fb => fb.rating === 'warning').length
  const goods    = pred.feedback.filter(fb => fb.rating === 'good').length

  return (
    <div className="min-h-screen bg-[#0d0d1a]">
      {/* Header */}
      <header className="border-b border-white/[0.06] px-6 py-4 flex items-center gap-4 sticky top-0 z-50 bg-[#0d0d1a]/95 backdrop-blur-xl">
        <button onClick={onReset} className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors text-sm">
          <ArrowLeft size={16} /> New analysis
        </button>
        <div className="w-px h-5 bg-white/10" />
        <div className="flex items-center gap-2">
          <span className="text-white font-bold">🏀 HoopSense AI</span>
          <span className="text-slate-600 text-xs">— Shot Analysis Report</span>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-slate-500 font-mono">{pred.session_id.slice(0, 8)}</span>
          <span className="text-xs bg-white/5 border border-white/10 px-2.5 py-1 rounded-full text-slate-400">
            v{pred.model_version}
          </span>
          <span className={`text-xs px-2.5 py-1 rounded-full font-semibold ${
            pred.confidence >= 0.8 ? 'bg-green-500/10 text-green-400 border border-green-500/20'
            : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
          }`}>
            {Math.round(pred.confidence * 100)}% confidence
          </span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Hero stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Form Score', value: Math.round(pred.form_score), unit: '/100', color: gradeColor(pred.form_grade),
              sub: pred.form_grade, icon: Activity },
            { label: 'Make Probability', value: Math.round(pred.make_probability * 100), unit: '%',
              color: pred.make_probability >= 0.6 ? '#22c55e' : pred.make_probability >= 0.4 ? '#f97316' : '#ef4444',
              sub: pred.make_probability >= 0.6 ? 'High' : pred.make_probability >= 0.4 ? 'Moderate' : 'Low', icon: Target },
            { label: 'Shot Arc', value: Math.round(pred.features.ball_arc_angle), unit: '°',
              color: '#3b82f6', sub: pred.features.ball_arc_angle >= 38 && pred.features.ball_arc_angle <= 55 ? 'Optimal' : 'Off target', icon: TrendingUp },
            { label: 'Distance', value: Math.round(pred.features.player_distance_hoop), unit: 'ft',
              color: '#a855f7', sub: 'from hoop', icon: Ruler },
          ].map(({ label, value, unit, color, sub, icon: Icon }) => (
            <div key={label} className="glass-card p-5">
              <div className="flex items-start justify-between mb-3">
                <p className="text-slate-400 text-xs font-medium uppercase tracking-wide">{label}</p>
                <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: `${color}20` }}>
                  <Icon size={14} style={{ color }} />
                </div>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-black leading-none" style={{ color }}>{value}</span>
                <span className="text-slate-400 text-sm">{unit}</span>
              </div>
              <p className="text-slate-500 text-xs mt-2">{sub}</p>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-white/[0.03] rounded-xl p-1 w-fit border border-white/[0.06]">
          {(['overview', 'biomechanics', 'feedback'] as const).map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`px-5 py-2 rounded-lg text-sm font-medium transition-all capitalize ${
                activeTab === tab ? 'bg-orange-500 text-white' : 'text-slate-400 hover:text-white'
              }`}>
              {tab}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left: Score + radar */}
            <div className="glass-card p-6 flex flex-col items-center gap-6">
              <div>
                <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wide text-center mb-4">Form Score</h3>
                <ScoreRing score={pred.form_score} grade={pred.form_grade} />
              </div>
              <div className="w-full border-t border-white/[0.06] pt-5">
                <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wide mb-3">Mechanics radar</h3>
                <RadarViz features={pred.features} />
              </div>
              <div className="w-full border-t border-white/[0.06] pt-5">
                <MakeGauge probability={pred.make_probability} />
              </div>
            </div>

            {/* Centre: Trajectory */}
            <div className="glass-card p-6">
              <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wide mb-4">Ball trajectory</h3>
              <TrajectoryViz trajectory={pred.ball_trajectory} hoop={pred.hoop_position} />
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="bg-white/[0.04] rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500 mb-1">Arc angle</p>
                  <p className="text-lg font-bold text-orange-400">{pred.features.ball_arc_angle.toFixed(1)}°</p>
                </div>
                <div className="bg-white/[0.04] rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500 mb-1">Smoothness</p>
                  <p className="text-lg font-bold text-blue-400">{Math.round(pred.features.trajectory_smoothness * 100)}%</p>
                </div>
              </div>
              {trajectoryVelocity.length > 2 && (
                <div className="mt-4">
                  <p className="text-xs text-slate-500 mb-2">Ball speed profile</p>
                  <ResponsiveContainer width="100%" height={80}>
                    <AreaChart data={trajectoryVelocity} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                      <defs>
                        <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <Area type="monotone" dataKey="speed" stroke="#f97316" fill="url(#speedGrad)" strokeWidth={1.5} dot={false} />
                      <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11 }} formatter={(v: number) => [`${v} px/s`, 'Speed']} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Right: Skeleton + phase breakdown */}
            <div className="space-y-4">
              <div className="glass-card p-6">
                <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wide mb-4">Pose overlay</h3>
                {pred.pose_frames.length > 0
                  ? <SkeletonViz poseFrames={pred.pose_frames} />
                  : <div className="h-40 flex items-center justify-center text-slate-600 text-sm">No pose data</div>
                }
              </div>

              <div className="glass-card p-5">
                <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wide mb-4">Shot phases</h3>
                <div className="space-y-2">
                  {Object.entries(pred.shot_phase_frames).map(([phase, count]) => {
                    const colors: Record<string, string> = {
                      gather: '#eab308', setup: '#3b82f6', release: '#f97316',
                      follow_through: '#22c55e', unknown: '#64748b'
                    }
                    const pct = Math.round((count / pred.total_frames) * 100)
                    const color = colors[phase] || '#64748b'
                    return (
                      <div key={phase}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="capitalize text-slate-400">{phase.replace('_', ' ')}</span>
                          <span className="text-slate-500">{count}f · {pct}%</span>
                        </div>
                        <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
                <div className="mt-4 pt-4 border-t border-white/[0.06] grid grid-cols-2 gap-3 text-xs text-slate-500">
                  <div><span className="text-slate-300">{pred.total_frames}</span> total frames</div>
                  <div><span className="text-slate-300">{pred.video_fps.toFixed(0)}</span> fps</div>
                  <div><span className="text-slate-300">{pred.video_duration_s.toFixed(1)}s</span> duration</div>
                  <div><span className="text-slate-300">{pred.features.shot_duration_frames}</span> shot frames</div>
                </div>
              </div>

              <div className="glass-card p-5">
                <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wide mb-3">Feedback summary</h3>
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-red-500/10 rounded-lg p-3">
                    <p className="text-2xl font-bold text-red-400">{errors}</p>
                    <p className="text-xs text-red-400/60 mt-1">Critical</p>
                  </div>
                  <div className="bg-amber-500/10 rounded-lg p-3">
                    <p className="text-2xl font-bold text-amber-400">{warnings}</p>
                    <p className="text-xs text-amber-400/60 mt-1">Warnings</p>
                  </div>
                  <div className="bg-green-500/10 rounded-lg p-3">
                    <p className="text-2xl font-bold text-green-400">{goods}</p>
                    <p className="text-xs text-green-400/60 mt-1">Good</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Biomechanics Tab */}
        {activeTab === 'biomechanics' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="text-white font-semibold mb-5">Joint angles & kinematics</h3>
              <div className="space-y-4">
                {biomechanicsData.map(({ name, value, unit, good }) => {
                  const inRange = value >= good[0] && value <= good[1]
                  const pct = unit === '%' ? value : Math.min(100, (value / (good[1] * 1.5)) * 100)
                  return (
                    <div key={name}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-sm text-slate-300">{name}</span>
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-mono font-semibold ${inRange ? 'text-green-400' : 'text-amber-400'}`}>
                            {value}{unit}
                          </span>
                          <span className="text-xs text-slate-600">({good[0]}–{good[1]}{unit})</span>
                        </div>
                      </div>
                      <div className="relative h-2 bg-white/[0.06] rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-700"
                          style={{ width: `${pct}%`, background: inRange ? '#22c55e' : '#f97316' }} />
                        <div className="absolute top-0 h-full w-px bg-green-500/40"
                          style={{ left: `${(good[0] / (good[1] * 1.5)) * 100}%` }} />
                        <div className="absolute top-0 h-full w-px bg-green-500/40"
                          style={{ left: `${(good[1] / (good[1] * 1.5)) * 100}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="space-y-6">
              <div className="glass-card p-6">
                <h3 className="text-white font-semibold mb-5">Timing analysis</h3>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: 'Shot duration', value: `${pred.features.shot_duration_frames}f`, sub: `${(pred.features.shot_duration_frames / pred.video_fps).toFixed(2)}s` },
                    { label: 'Release timing', value: `${Math.round(pred.features.release_timing_score * 100)}%`, sub: 'at jump peak' },
                    { label: 'Rhythm score', value: `${Math.round(pred.features.rhythm_consistency * 100)}%`, sub: 'gather → release' },
                    { label: 'Jump height', value: `${Math.round(pred.features.jump_height_normalized * 100)}%`, sub: 'normalized' },
                  ].map(({ label, value, sub }) => (
                    <div key={label} className="bg-white/[0.04] rounded-xl p-4 text-center">
                      <p className="text-xs text-slate-500 mb-1">{label}</p>
                      <p className="text-2xl font-bold text-white">{value}</p>
                      <p className="text-xs text-slate-600 mt-1">{sub}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-card p-6">
                <h3 className="text-white font-semibold mb-5">Ball flight data</h3>
                <div className="space-y-3">
                  {[
                    { label: 'Arc angle', value: `${pred.features.ball_arc_angle.toFixed(1)}°`, ideal: '38–55°', ok: pred.features.ball_arc_angle >= 38 && pred.features.ball_arc_angle <= 55 },
                    { label: 'Release velocity', value: `${(pred.features.ball_velocity_release * 1000).toFixed(1)} px/s`, ideal: 'Higher = more power', ok: true },
                    { label: 'Peak velocity', value: `${(pred.features.ball_velocity_peak * 1000).toFixed(1)} px/s`, ideal: 'Launch phase', ok: true },
                    { label: 'Trajectory smoothness', value: `${Math.round(pred.features.trajectory_smoothness * 100)}%`, ideal: '>80%', ok: pred.features.trajectory_smoothness >= 0.8 },
                    { label: 'Release height', value: `${Math.round(pred.features.release_height_normalized * 100)}%`, ideal: '>60%', ok: pred.features.release_height_normalized >= 0.6 },
                  ].map(({ label, value, ideal, ok }) => (
                    <div key={label} className="flex items-center justify-between py-2 border-b border-white/[0.04] last:border-0">
                      <div>
                        <p className="text-sm text-slate-300">{label}</p>
                        <p className="text-xs text-slate-600">{ideal}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-mono font-semibold text-white">{value}</span>
                        {ok ? <CheckCircle2 size={14} className="text-green-400" /> : <AlertTriangle size={14} className="text-amber-400" />}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Feedback Tab */}
        {activeTab === 'feedback' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-3">
              <h3 className="text-white font-semibold text-lg">Biomechanical feedback</h3>
              <div className="space-y-2.5">
                {pred.feedback.map((fb, i) => <FeedbackCard key={i} item={fb} />)}
              </div>
            </div>

            <div className="space-y-4">
              <div className="glass-card p-6">
                <h3 className="text-white font-semibold mb-5">Overall assessment</h3>
                <ScoreRing score={pred.form_score} grade={pred.form_grade} />
                <div className="mt-5 space-y-2 text-sm">
                  <p className="text-slate-300 leading-relaxed">
                    {pred.form_score >= 85
                      ? '🏆 Elite-level mechanics. Minor refinements only. Your form is consistent with professional shooting technique.'
                      : pred.form_score >= 70
                      ? '✅ Solid fundamentals with room for improvement in 1-2 areas. Focus on the warnings above.'
                      : pred.form_score >= 50
                      ? '⚠️ Average mechanics. Several areas need attention — prioritise the critical issues first.'
                      : '🔴 Significant mechanical issues detected. Work with a coach to address the fundamentals before individual adjustments.'}
                  </p>
                </div>
              </div>

              <div className="glass-card p-6">
                <h3 className="text-white font-semibold mb-4">Priority improvements</h3>
                <div className="space-y-3">
                  {pred.feedback.filter(fb => fb.rating !== 'good').slice(0, 3).map((fb, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span className="w-5 h-5 rounded-full bg-orange-500/20 text-orange-400 text-xs flex items-center justify-center flex-shrink-0 font-bold mt-0.5">{i+1}</span>
                      <div>
                        <p className="text-sm text-slate-300 font-medium">{fb.aspect}</p>
                        <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{fb.message.slice(0, 80)}...</p>
                      </div>
                    </div>
                  ))}
                  {pred.feedback.filter(fb => fb.rating !== 'good').length === 0 && (
                    <p className="text-green-400 text-sm">🎯 No major issues — your form is excellent!</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
