import { useEffect, useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import { cn } from '@/lib/utils'

export function LiveMonitor() {
  const [isRecording, setIsRecording] = useState(true)
  const [videoSrc, setVideoSrc] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file)
      setError('')
      setVideoSrc(url)
    } else {
      setError('Unsupported file. Please upload a valid video.')
    }
  }

  useEffect(() => {
    return () => {
      if (videoSrc) URL.revokeObjectURL(videoSrc)
    }
  }, [videoSrc])

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files[0]
    handleFileSelect(file)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  return (
    <div className="flex h-full flex-col gap-6">
      <div
        className={cn(
          'relative flex-1 overflow-hidden rounded-xl border bg-card transition-colors',
          isDragging ? 'border-accent' : 'border-border',
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {videoSrc ? (
          <>
            <video
              ref={videoRef}
              src={videoSrc}
              autoPlay
              loop
              muted
              playsInline
              controls
              onError={() =>
                setError('Unable to play this video. Try an H.264 MP4 file.')
              }
              className="absolute inset-0 h-full w-full object-cover"
            />

            <div className="absolute left-4 top-4 rounded-md border border-border bg-background/80 px-2.5 py-1">
              <span className="font-mono text-xs text-foreground">
                Channel 01
              </span>
            </div>

            <div className="absolute right-4 top-4 flex items-center gap-3">
              <div className="flex items-center gap-1.5 rounded-md border border-border bg-background/80 px-2 py-1">
                <div
                  className={cn(
                    'h-2 w-2 rounded-full',
                    isRecording ? 'bg-rose-500 animate-pulse' : 'bg-muted-foreground',
                  )}
                />
                <span className="text-[10px] font-medium text-muted-foreground">
                  REC
                </span>
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded-md border border-border bg-background/80 p-1.5 transition-colors hover:bg-background"
              >
                <Upload className="h-3.5 w-3.5 text-muted-foreground" />
              </button>
            </div>
            {error && (
              <div className="absolute bottom-4 left-4 rounded-md border border-rose-200/60 bg-rose-50 px-3 py-2 text-xs text-rose-700">
                {error}
              </div>
            )}
          </>
        ) : (
          <>
            <div className="absolute inset-0 flex items-center justify-center bg-muted/40">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer rounded-lg border border-dashed border-border bg-background/70 p-8 text-center transition-colors hover:border-accent/50 hover:bg-background"
              >
                <Upload className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
                <p className="text-sm font-medium text-foreground">
                  Drop video or click to upload
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  MP4, WebM, MOV
                </p>
              </button>
            </div>

            <div className="absolute left-4 top-4 rounded-md border border-border bg-background/80 px-2.5 py-1">
              <span className="font-mono text-xs text-muted-foreground">
                No Feed
              </span>
            </div>
            {error && (
              <div className="absolute bottom-4 left-4 rounded-md border border-rose-200/60 bg-rose-50 px-3 py-2 text-xs text-rose-700">
                {error}
              </div>
            )}
          </>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={(event) =>
            event.target.files?.[0] && handleFileSelect(event.target.files[0])
          }
          className="hidden"
        />
      </div>
    </div>
  )
}
