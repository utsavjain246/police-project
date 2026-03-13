import { cn } from '@/lib/utils'
import { Shield, Video, Search, Clock } from 'lucide-react'
import { ThemeSwitch } from '@/components/theme-switch'

const formatDurationMs = (value) => {
  const ms = Number(value)
  if (!Number.isFinite(ms) || ms < 0) return '--'
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  return `${Math.round(ms)}ms`
}

export function Sidebar({ mode = 'monitoring', onModeChange, showChannels = true, timingStats }) {
  const modeLinkClass = (isActive) =>
    cn(
      'flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-xs font-medium transition-colors',
      isActive
        ? 'bg-background text-foreground shadow-sm'
        : 'text-muted-foreground hover:bg-background/60 hover:text-foreground',
    )

  const latencyCards = timingStats
    ? [
        {
          label: 'Frontend post-upload to answer',
          value: formatDurationMs(timingStats.client_post_upload_to_answer_ms),
        },
        {
          label: 'Frontend end-to-end',
          value: formatDurationMs(timingStats.client_end_to_end_ms),
        },
        {
          label: 'Frontend upload',
          value: formatDurationMs(timingStats.client_upload_ms),
        },
        {
          label: 'Backend post-upload to complete',
          value: formatDurationMs(timingStats.backend_post_upload_to_complete_ms),
        },
        {
          label: 'Backend processing only',
          value: formatDurationMs(timingStats.backend_processing_ms),
        },
        {
          label: 'Backend queue wait',
          value: formatDurationMs(timingStats.backend_queue_wait_ms),
        },
      ]
    : []

  return (
    <aside className="flex w-64 flex-col border-r border-border bg-card">
      <div className="flex items-center gap-4 border-b border-border px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent/20 bg-accent/10">
          <Shield className="h-4 w-4 text-accent" />
        </div>
        <div>
          <div className="text-sm font-semibold">AI Surveillance System</div>
          <div className="text-xs text-muted-foreground">Ops Monitor</div>
        </div>
      </div>

      <div className="border-b border-border px-3 py-4">
        <div className="flex gap-1 rounded-lg bg-muted p-1">
          <button
            type="button"
            onClick={() => onModeChange?.('monitoring')}
            className={modeLinkClass(mode === 'monitoring')}
          >
            <Video className="h-3.5 w-3.5" />
            Monitoring
          </button>
          <button
            type="button"
            onClick={() => onModeChange?.('forensics')}
            className={modeLinkClass(mode === 'forensics')}
          >
            <Search className="h-3.5 w-3.5" />
            Forensics
          </button>
        </div>
      </div>

      <div className="flex-1 p-3">
        {showChannels && (
          <>
            <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Channels
            </p>
            <div className="space-y-2">
              {['Channel 01', 'Channel 02', 'Channel 03', 'Channel 04'].map(
                (channel, i) => (
                  <button
                    key={channel}
                    type="button"
                    className={cn(
                      'flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors',
                      i === 0
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
                    )}
                  >
                    <div
                      className={cn(
                        'h-2 w-2 rounded-full',
                        i === 0 ? 'bg-emerald-500' : 'bg-muted-foreground/40',
                      )}
                    />
                    {channel}
                  </button>
                ),
              )}
            </div>
          </>
        )}
        {!showChannels && timingStats && (
          <div className="mt-4">
            <div className="mb-3 flex items-center gap-2 px-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              <Clock className="h-3.5 w-3.5" />
              Last Run Latency
            </div>
            <div className="space-y-2">
              {latencyCards.map((item) => (
                <div
                  key={item.label}
                  className="rounded-lg border border-white/10 bg-black/40 px-3 py-2"
                >
                  <p className="text-[11px] text-muted-foreground">{item.label}</p>
                  <p className="mt-1 text-sm font-semibold text-foreground">{item.value}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="border-t border-border p-4">
        <ThemeSwitch />
      </div>
    </aside>
  )
}
