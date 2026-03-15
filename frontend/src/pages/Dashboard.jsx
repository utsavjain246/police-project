import { useState } from 'react'
import { Activity, Clock, TrendingUp, Zap } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Navigation } from '@/components/navigation'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export default function DashboardPage() {
  const [timeRange, setTimeRange] = useState('12h')

  const stats = [
    {
      title: 'Total Operations',
      value: '2,847',
      change: '+12.5%',
      delta: 12.5,
      icon: Activity,
      tone: 'sky',
      spark: [22, 38, 30, 46, 36, 52, 40, 62, 48, 70],
    },
    {
      title: 'Success Rate',
      value: '99.2%',
      change: '+0.8%',
      delta: 0.8,
      icon: TrendingUp,
      tone: 'emerald',
      spark: [88, 92, 90, 96, 94, 97, 96, 98, 97, 99],
    },
    {
      title: 'Avg Response Time',
      value: '142ms',
      change: '-8.3%',
      delta: -8.3,
      icon: Zap,
      tone: 'amber',
      spark: [60, 58, 62, 54, 50, 55, 48, 46, 52, 45],
    },
    {
      title: 'Active Jobs',
      value: '24',
      change: '+4',
      delta: 4,
      icon: Clock,
      tone: 'violet',
      spark: [12, 16, 14, 18, 20, 22, 18, 24, 26, 24],
    },
  ]

  const toneStyles = {
    sky: {
      icon: 'text-sky-300',
      glow: 'from-sky-500/25 via-transparent to-transparent',
      bar: 'bg-sky-400/80',
      accent: 'bg-sky-500/10 text-sky-200',
    },
    emerald: {
      icon: 'text-emerald-300',
      glow: 'from-emerald-500/25 via-transparent to-transparent',
      bar: 'bg-emerald-400/80',
      accent: 'bg-emerald-500/10 text-emerald-200',
    },
    amber: {
      icon: 'text-amber-300',
      glow: 'from-amber-500/25 via-transparent to-transparent',
      bar: 'bg-amber-400/80',
      accent: 'bg-amber-500/10 text-amber-200',
    },
    violet: {
      icon: 'text-violet-300',
      glow: 'from-violet-500/25 via-transparent to-transparent',
      bar: 'bg-violet-400/80',
      accent: 'bg-violet-500/10 text-violet-200',
    },
  }

  const recentJobs = [
    {
      id: 'job_8d7a9b2c',
      name: 'CSV Transform',
      status: 'completed',
      time: '2 min ago',
      duration: '1.2s',
    },
    {
      id: 'job_3f4e5a1b',
      name: 'JSON Processing',
      status: 'completed',
      time: '5 min ago',
      duration: '0.8s',
    },
    {
      id: 'job_6c2d8e9f',
      name: 'Data Validation',
      status: 'running',
      time: 'Just now',
      duration: '3.1s',
    },
    {
      id: 'job_1a5b7c4d',
      name: 'XML Parser',
      status: 'completed',
      time: '12 min ago',
      duration: '2.4s',
    },
    {
      id: 'job_9e3f2a8c',
      name: 'API Integration',
      status: 'failed',
      time: '18 min ago',
      duration: '0.3s',
    },
  ]

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900/70 to-black text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(900px_circle_at_12%_0%,rgba(56,189,248,0.12),transparent_55%),radial-gradient(750px_circle_at_88%_10%,rgba(249,115,22,0.1),transparent_55%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:44px_44px] opacity-30" />
      <Navigation />
      <main className="relative z-10 mx-auto max-w-6xl px-6 py-12 sm:px-8 lg:px-10">
        <div className="mb-10 flex flex-wrap items-center justify-between gap-6">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              DataFlow Control
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight">Operations Pulse</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Live throughput, reliability, and workload pressure at a glance.
            </p>
          </div>
          <div className="flex items-center gap-3 rounded-full border border-white/10 bg-white/5 px-4 py-2">
            <span className="text-xs text-muted-foreground">Window</span>
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="h-8 w-32 border-white/10 bg-black/60 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">Last hour</SelectItem>
                <SelectItem value="12h">Last 12 hours</SelectItem>
                <SelectItem value="24h">Last 24 hours</SelectItem>
                <SelectItem value="7d">Last 7 days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="mb-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => {
            const Icon = stat.icon
            const tone = toneStyles[stat.tone] || toneStyles.sky
            const deltaClass = stat.delta >= 0 ? 'text-emerald-300' : 'text-rose-300'
            return (
              <Card
                key={stat.title}
                className="group relative overflow-hidden border border-white/10 bg-black/60 shadow-[0_20px_50px_rgba(0,0,0,0.35)]"
              >
                <div
                  className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${tone.glow} opacity-0 transition duration-300 group-hover:opacity-100`}
                />
                <CardHeader className="relative flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    {stat.title}
                  </CardTitle>
                  <span className={`rounded-full px-2 py-1 text-[10px] ${tone.accent}`}>
                    Live
                  </span>
                </CardHeader>
                <CardContent className="relative space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-semibold">{stat.value}</div>
                    <Icon className={`h-5 w-5 ${tone.icon}`} />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className={deltaClass}>{stat.change}</span>
                    <span>vs prior window</span>
                  </div>
                  <div className="flex h-8 items-end gap-1">
                    {stat.spark.map((height, idx) => (
                      <div
                        key={`${stat.title}-${idx}`}
                        className={`flex-1 rounded-t ${tone.bar}`}
                        style={{ height: `${height}%` }}
                      />
                    ))}
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>

        <div className="mb-10 grid gap-6 lg:grid-cols-2">
          <Card className="overflow-hidden border border-white/10 bg-black/60 shadow-[0_20px_50px_rgba(0,0,0,0.35)]">
            <CardHeader>
              <CardTitle className="text-base font-semibold">Operations Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative h-64 overflow-hidden rounded-lg border border-white/10 bg-gradient-to-br from-slate-950/80 via-black/70 to-black">
                <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:28px_28px] opacity-40" />
                <div className="relative flex h-full items-end justify-around p-4">
                  {[65, 78, 52, 88, 72, 95, 68, 82, 76, 90, 85, 92].map(
                    (height, i) => (
                      <div
                        key={`operations-${i}`}
                        className="w-6 rounded-t-full bg-gradient-to-t from-sky-500/80 via-sky-300/60 to-sky-200/30 shadow-[0_0_12px_rgba(56,189,248,0.4)]"
                        style={{ height: `${height}%` }}
                      />
                    ),
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden border border-white/10 bg-black/60 shadow-[0_20px_50px_rgba(0,0,0,0.35)]">
            <CardHeader>
              <CardTitle className="text-base font-semibold">Response Times</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative h-64 overflow-hidden rounded-lg border border-white/10 bg-gradient-to-br from-slate-950/80 via-black/70 to-black">
                <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:28px_28px] opacity-40" />
                <div className="relative flex h-full items-end justify-around p-4">
                  {[45, 52, 38, 62, 48, 55, 42, 58, 50, 65, 52, 60].map(
                    (height, i) => (
                      <div
                        key={`response-${i}`}
                        className="w-6 rounded-t-full bg-gradient-to-t from-amber-500/80 via-amber-300/60 to-amber-200/30 shadow-[0_0_12px_rgba(251,191,36,0.4)]"
                        style={{ height: `${height}%` }}
                      />
                    ),
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="border border-white/10 bg-black/60 shadow-[0_20px_50px_rgba(0,0,0,0.35)]">
          <CardHeader>
            <CardTitle className="text-base font-semibold">Recent Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {recentJobs.map((job) => (
                <div
                  key={job.id}
                  className="flex flex-wrap items-center justify-between gap-6 border-b border-white/10 pb-6 last:border-0 last:pb-0"
                >
                  <div className="flex items-center gap-4">
                    <div
                      className={`h-10 w-1 rounded-full ${
                        job.status === 'completed'
                          ? 'bg-emerald-400/80'
                          : job.status === 'running'
                            ? 'bg-sky-400/80'
                            : 'bg-rose-400/80'
                      }`}
                    />
                    <div>
                      <div className="font-medium">{job.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {job.id}
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-4 text-sm">
                    <div className="text-muted-foreground">{job.time}</div>
                    <div className="font-mono text-muted-foreground">
                      {job.duration}
                    </div>
                    <div
                      className={`min-w-20 rounded-full border border-white/10 px-3 py-1 text-center text-xs font-medium ${
                        job.status === 'completed'
                          ? 'bg-emerald-500/10 text-emerald-200'
                          : job.status === 'running'
                            ? 'bg-sky-500/10 text-sky-200'
                            : 'bg-rose-500/10 text-rose-200'
                      }`}
                    >
                      {job.status}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
