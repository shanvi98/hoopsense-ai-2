export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed'
export type FormGrade = 'Elite' | 'Good' | 'Average' | 'Poor'
export type ShotPhase = 'setup' | 'gather' | 'release' | 'follow_through' | 'unknown'
export type FeedbackRating = 'good' | 'warning' | 'error'

export interface BiomechanicalFeatures {
  elbow_angle_release: number; elbow_angle_min: number; knee_bend_max: number
  wrist_release_angle: number; hip_shoulder_alignment: number; trunk_lean_angle: number
  shooting_arm_extension: number; release_height_normalized: number; ball_arc_angle: number
  ball_velocity_release: number; ball_velocity_peak: number; trajectory_smoothness: number
  shot_duration_frames: number; jump_height_normalized: number; release_timing_score: number
  rhythm_consistency: number; player_distance_hoop: number; shot_angle_degrees: number
  balance_score: number; follow_through_score: number
}

export interface FeedbackItem {
  aspect: string; rating: FeedbackRating; message: string
  metric_value?: number; benchmark_range?: string
}

export interface TrajectoryPoint { x: number; y: number; frame: number; vx: number; vy: number }
export interface HoopPosition { x: number; y: number; w: number; h: number }
export interface PoseLandmark { x: number; y: number; visibility: number }

export interface PoseFrame {
  frame_idx: number; timestamp_ms: number; phase: ShotPhase
  landmarks: Record<string, PoseLandmark>; ball?: { x: number; y: number } | null
}

export interface ShotPrediction {
  session_id: string; status: AnalysisStatus; form_score: number; form_grade: FormGrade
  make_probability: number; confidence: number; features: BiomechanicalFeatures
  feedback: FeedbackItem[]; ball_trajectory: TrajectoryPoint[]; pose_frames: PoseFrame[]
  hoop_position: HoopPosition | null; video_fps: number; video_duration_s: number
  total_frames: number; shot_phase_frames: Record<string, number>; model_version: string
}

export interface AnalysisResult {
  session_id: string; status: AnalysisStatus; result?: ShotPrediction
  error?: string; processing_time_ms: number
}
