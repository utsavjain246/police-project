import { useEffect, useState } from 'react'

export function TelemetryPanel() {
  const [fps, setFps] = useState(30)
  const [latency, setLatency] = useState(45)

  useEffect(() => {
    const interval = setInterval(() => {
      setFps(Math.floor(28 + Math.random() * 4))
      setLatency(Math.floor(40 + Math.random() * 15))
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="h-11 border-t border-border bg-muted/70 px-4">
      <div className="flex h-full items-center justify-between text-xs">
        <div className="flex items-center gap-4 font-mono text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span>
              Status: <span className="text-emerald-600">Online</span>
            </span>
          </div>
          <span className="text-muted-foreground/60">|</span>
          <span>
            FPS: <span className="text-foreground">{fps}</span>
          </span>
          <span className="text-muted-foreground/60">|</span>
          <span>
            Latency: <span className="text-foreground">{latency}ms</span>
          </span>
        </div>
        <span className="text-xs text-muted-foreground">DataFlow v1.0</span>
      </div>
    </div>
  )
}
