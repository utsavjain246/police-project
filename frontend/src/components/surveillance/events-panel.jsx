import { useEffect, useState } from 'react'

const mockEvents = [
  {
    id: 1,
    time: '14:32:45',
    type: 'Motion Detected',
    severity: 'low',
    summary: 'Unusual movement near loading bay.',
    evidenceType: 'clip',
  },
  {
    id: 2,
    time: '14:31:22',
    type: 'Person Detected',
    severity: 'low',
    summary: 'Single person entered restricted corridor.',
    evidenceType: 'snapshot',
  },
  {
    id: 3,
    time: '14:30:58',
    type: 'Crowd Gathering',
    severity: 'medium',
    summary: 'Crowd density exceeded normal threshold.',
    evidenceType: 'gif',
  },
  {
    id: 4,
    time: '14:28:15',
    type: 'Person Detected',
    severity: 'low',
    summary: 'Authorized staff member detected.',
    evidenceType: 'timestamp',
  },
  {
    id: 5,
    time: '14:25:33',
    type: 'Motion Detected',
    severity: 'low',
    summary: 'Brief motion in outer perimeter.',
    evidenceType: 'gif',
  },
  {
    id: 6,
    time: '14:22:10',
    type: 'Vehicle Stopped',
    severity: 'medium',
    summary: 'Vehicle stopped longer than usual.',
    evidenceType: 'clip',
  },
  {
    id: 7,
    time: '14:20:45',
    type: 'Person Detected',
    severity: 'low',
    summary: 'Entry detected at service door.',
    evidenceType: 'snapshot',
  },
  {
    id: 8,
    time: '14:18:30',
    type: 'Motion Detected',
    severity: 'low',
    summary: 'Movement detected in stairwell.',
    evidenceType: 'timestamp',
  },
  {
    id: 9,
    time: '14:15:22',
    type: 'Person Detected',
    severity: 'low',
    summary: 'Repeated pass through checkpoint.',
    evidenceType: 'gif',
  },
  {
    id: 10,
    time: '14:12:08',
    type: 'Motion Detected',
    severity: 'low',
    summary: 'Minor motion detected near entrance.',
    evidenceType: 'clip',
  },
]

export function EventsPanel({ onSelectEvent, selectedEventId }) {
  const [currentDate, setCurrentDate] = useState('')

  useEffect(() => {
    const now = new Date()
    setCurrentDate(
      now.toLocaleDateString('en-US', {
        month: 'short',
        day: '2-digit',
        year: 'numeric',
      }),
    )
  }, [])

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high':
        return 'text-rose-600'
      case 'medium':
        return 'text-amber-600'
      default:
        return 'text-muted-foreground'
    }
  }

  return (
    <aside className="flex w-64 flex-col border-l border-border bg-card">
      <div className="border-b border-border px-4 py-3">
        <h3 className="text-sm font-medium text-foreground">Recent Events</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {currentDate}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {mockEvents.map((event, index) => (
          <button
            key={event.id}
            type="button"
            onClick={() => onSelectEvent?.(event)}
            className={`w-full cursor-pointer px-4 py-3 text-left transition-colors hover:bg-muted/60 ${
              index !== mockEvents.length - 1
                ? 'border-b border-border'
                : ''
            } ${selectedEventId === event.id ? 'bg-muted/70' : ''}`}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-xs font-mono text-muted-foreground">
                {event.time}
              </span>
              <span className={`text-xs text-right ${getSeverityColor(event.severity)}`}>
                {event.type}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{event.summary}</p>
          </button>
        ))}
      </div>
    </aside>
  )
}
