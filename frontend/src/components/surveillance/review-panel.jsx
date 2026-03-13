import { useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

export function ReviewPanel({ event, notes, onNotesChange, onClose, onDecision }) {
  const evidenceLabel = useMemo(() => {
    if (!event) return ''
    switch (event.evidenceType) {
      case 'snapshot':
        return 'Snapshot Evidence'
      case 'gif':
        return 'GIF Evidence'
      case 'clip':
        return 'Video Clip Evidence'
      default:
        return 'Timestamp Evidence'
    }
  }, [event])

  if (!event) return null

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/30 p-4 sm:items-center">
      <div className="w-full max-w-3xl overflow-hidden rounded-2xl border border-border bg-card shadow-soft">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
              Human Review
            </p>
            <h2 className="text-lg font-semibold">{event.type}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-border px-3 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
          >
            Close
          </button>
        </div>

        <div className="grid gap-6 p-6 lg:grid-cols-[1.3fr_1fr]">
          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-muted/30 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    Evidence
                  </p>
                  <p className="text-sm font-medium">{evidenceLabel}</p>
                </div>
                <span className="text-xs text-muted-foreground">
                  {event.timestamp}
                </span>
              </div>
              <div className="mt-4">
                {event.evidenceType === 'snapshot' && (
                  <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-border bg-gradient-to-br from-muted/40 via-muted/10 to-background">
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                      Snapshot preview
                    </div>
                  </div>
                )}
                {event.evidenceType === 'gif' && (
                  <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-border bg-gradient-to-br from-muted/50 via-muted/20 to-background">
                    <div className="absolute left-3 top-3 rounded-full bg-black/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-white">
                      GIF
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                      Animated evidence preview
                    </div>
                  </div>
                )}
                {event.evidenceType === 'clip' && (
                  <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-border bg-gradient-to-br from-muted/50 via-muted/20 to-background">
                    <div className="absolute left-3 top-3 rounded-full bg-black/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-white">
                      Clip
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                      Video clip preview
                    </div>
                  </div>
                )}
                {event.evidenceType !== 'snapshot' &&
                  event.evidenceType !== 'gif' &&
                  event.evidenceType !== 'clip' && (
                    <div className="rounded-lg border border-border bg-background px-4 py-6 text-center text-sm text-muted-foreground">
                      Detection occurred at {event.timestamp}
                    </div>
                  )}
              </div>
            </div>

            <div className="rounded-xl border border-border bg-background p-4">
              <p className="text-sm font-medium">Summary</p>
              <p className="mt-2 text-sm text-muted-foreground">
                {event.summary}
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">Reviewer Notes</p>
                <Button variant="outline" size="sm" onClick={() => onDecision('notes')}>
                  Add Notes
                </Button>
              </div>
              <Textarea
                value={notes}
                onChange={(event) => onNotesChange(event.target.value)}
                className="mt-3 h-40"
                placeholder="Add your notes or context..."
              />
              <p className="mt-2 text-xs text-muted-foreground">
                Notes are saved with the decision log.
              </p>
            </div>

            <div className="rounded-xl border border-border bg-background p-4">
              <p className="text-sm font-medium">Decision</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button onClick={() => onDecision('approved')}>Approve</Button>
                <Button variant="destructive" onClick={() => onDecision('rejected')}>
                  Reject
                </Button>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                Decisions are recorded with your notes and timestamp.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
