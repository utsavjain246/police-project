import { useState } from 'react'
import { Sidebar } from '@/components/surveillance/sidebar'
import { LiveMonitor } from '@/components/surveillance/live-monitor'
import { TelemetryPanel } from '@/components/surveillance/telemetry-panel'
import { ForensicLab } from '@/components/surveillance/forensic-lab'

export default function Home() {
  const [mode, setMode] = useState('monitoring')
  const [forensicTimingStats, setForensicTimingStats] = useState(null)
  const isForensics = mode === 'forensics'
  const handleModeChange = (nextMode) => {
    setMode(nextMode)
    if (nextMode !== 'forensics') {
      setForensicTimingStats(null)
    }
  }

  return (
    <div
      className={`h-screen ${isForensics ? 'bg-black text-foreground forensic-theme' : 'bg-background'}`}
    >
      <div className="flex h-full">
        <Sidebar
          mode={mode}
          onModeChange={handleModeChange}
          showChannels={!isForensics}
          timingStats={isForensics ? forensicTimingStats : null}
        />

        <div className="flex min-w-0 flex-1 flex-col">
          <main className="flex-1 overflow-auto p-6 lg:p-8">
            {isForensics ? (
              <ForensicLab onTimingStats={setForensicTimingStats} />
            ) : (
              <LiveMonitor />
            )}
          </main>
          {!isForensics && <TelemetryPanel />}
        </div>

      </div>

    </div>
  )
}
