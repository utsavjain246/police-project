import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile } from '@ffmpeg/util'
import coreURL from '@ffmpeg/core?url'
import wasmURL from '@ffmpeg/core/wasm?url'
import {
  AlertTriangle,
  Car,
  Clock,
  Crosshair,
  Database,
  Download,
  FileVideo,
  ImageIcon,
  Pause,
  Play,
  RefreshCw,
  Scissors,
  Search,
  Upload,
  Users,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const SEGMENT_STEP = 0.04
const SEGMENT_MIN_GAP = 0.2
const SEGMENT_EPSILON = 0.08
const DEFAULT_TIMEOUT_MS = 60_000
const STATUS_TIMEOUT_MS = 45_000
const PREVIEW_TIMEOUT_MS = 300_000
const UPLOAD_TIMEOUT_MS = 1_800_000
const RETRYABLE_STATUSES = new Set([502, 503, 504])
const FAST_TRIM_MODE = true
const MAGIC_SCAN_STEPS = {
  general: [
    { at: 0, title: 'Preparing video', detail: 'Aligning the input stream' },
    { at: 10, title: 'Sampling frames', detail: 'Sweeping across the timeline' },
    { at: 32, title: 'Extracting signals', detail: 'Distilling patterns from noise' },
    { at: 58, title: 'Cross-linking events', detail: 'Aligning temporal echoes' },
    { at: 80, title: 'Verifying results', detail: 'Stabilizing the signal lattice' },
    { at: 94, title: 'Finalizing report', detail: 'Sealing the evidence stream' },
  ],
  face: [
    { at: 0, title: 'Preparing video', detail: 'Aligning the reference portrait' },
    { at: 12, title: 'Sampling frames', detail: 'Searching for candidate faces' },
    { at: 36, title: 'ArcFace screening', detail: 'Computing facial embeddings' },
    { at: 62, title: 'Gemini verification', detail: 'Confirming visual match' },
    { at: 84, title: 'Consolidating matches', detail: 'Filtering duplicates' },
    { at: 96, title: 'Finalizing report', detail: 'Sealing the evidence stream' },
  ],
}

const describeFetchError = (err) => {
  if (err?.name === 'AbortError') {
    return 'Request timed out. The server may be slow or unreachable.'
  }
  if (err?.message && err.message.toLowerCase().includes('failed to fetch')) {
    return `Network error: Unable to reach the API at ${API_BASE_URL}. Check that the backend is running, CORS is allowed, and HTTPS/Mixed Content is not blocking the request.`
  }
  return err?.message || 'Network error occurred.'
}

const isRetryableError = (err) => {
  if (err?.status && RETRYABLE_STATUSES.has(err.status)) {
    return true
  }
  const msg = String(err?.message || '').toLowerCase()
  return (
    msg.includes('timed out') ||
    msg.includes('failed to fetch') ||
    msg.includes('network error') ||
    msg.includes('unreachable')
  )
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const fetchWithTimeout = async (url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) => {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    throw new Error('You appear to be offline. Check your network connection.')
  }
  const externalSignal = options.signal
  if (!timeoutMs || timeoutMs <= 0) {
    try {
      return await fetch(url, options)
    } catch (err) {
      throw new Error(describeFetchError(err))
    }
  }
  const controller = new AbortController()
  let abortedByUser = false
  const handleAbort = () => {
    abortedByUser = true
    controller.abort()
  }
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort()
    } else {
      externalSignal.addEventListener('abort', handleAbort, { once: true })
    }
  }
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (err) {
    if (abortedByUser) {
      throw new Error('Request canceled.')
    }
    throw new Error(describeFetchError(err))
  } finally {
    clearTimeout(timeout)
    if (externalSignal) {
      externalSignal.removeEventListener('abort', handleAbort)
    }
  }
}

const uploadWithProgress = (url, formData, { timeoutMs, onProgress, onStart } = {}) =>
  new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.timeout = timeoutMs || 0
    if (onStart) onStart(xhr)

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return
      const pct = Math.round((event.loaded / event.total) * 100)
      if (onProgress) onProgress(pct)
    }

    xhr.onload = () => {
      const status = xhr.status
      const text = xhr.responseText || ''
      if (status >= 200 && status < 300) {
        resolve(text)
        return
      }
      const err = new Error(text || `Request failed (${status})`)
      err.status = status
      reject(err)
    }

    xhr.onerror = () => {
      reject(new Error('Network error occurred.'))
    }

    xhr.ontimeout = () => {
      reject(new Error('Request timed out. The server may be slow or unreachable.'))
    }

    xhr.onabort = () => {
      reject(new Error('Upload canceled.'))
    }

    xhr.send(formData)
  })

const fetchWithRetry = async (
  url,
  options = {},
  {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    retries = 2,
    retryDelayMs = 1000,
    retryStatuses = RETRYABLE_STATUSES,
    onRetry = null,
  } = {}
) => {
  let attempt = 0
  while (true) {
    try {
      const resp = await fetchWithTimeout(url, options, timeoutMs)
      if (retryStatuses.has(resp.status) && attempt < retries) {
        attempt += 1
        if (onRetry) {
          onRetry(attempt, resp)
        }
        await sleep(retryDelayMs * Math.pow(2, attempt - 1))
        continue
      }
      return resp
    } catch (err) {
      if (attempt >= retries || !isRetryableError(err)) {
        throw err
      }
      attempt += 1
      if (onRetry) {
        onRetry(attempt, err)
      }
      await sleep(retryDelayMs * Math.pow(2, attempt - 1))
    }
  }
}

const readJsonResponse = async (resp) => {
  const text = await resp.text()
  if (!resp.ok) {
    throw new Error(text || `Request failed (${resp.status})`)
  }
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch (err) {
    throw new Error('Invalid JSON response from server.')
  }
}

const tryParseJson = (value) => {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  if (!trimmed) return value
  const firstChar = trimmed[0]
  if (firstChar !== '{' && firstChar !== '[') return value
  try {
    return JSON.parse(trimmed)
  } catch (err) {
    return value
  }
}

const formatTimestamp = (seconds, { showMs = false } = {}) => {
  const numeric =
    typeof seconds === 'number' ? seconds : seconds == null ? NaN : Number(seconds)
  if (seconds === undefined || seconds === null || Number.isNaN(numeric)) {
    return showMs ? '--:--:--.---' : '--:--:--'
  }
  const totalMs = Math.max(0, Math.round(numeric * 1000))
  const hrs = String(Math.floor(totalMs / 3600000)).padStart(2, '0')
  const mins = String(Math.floor((totalMs % 3600000) / 60000)).padStart(2, '0')
  const secs = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, '0')
  const ms = String(totalMs % 1000).padStart(3, '0')
  const base = `${hrs}:${mins}:${secs}`
  return showMs ? `${base}.${ms}` : base
}

const parseTimestamp = (value) => {
  if (typeof value === 'number') return Number(value)
  if (!value || typeof value !== 'string') return null
  const raw = value.trim()
  if (!raw) return null
  const normalized = raw.replace(',', '.')
  const parts = normalized.split(':').map((part) => part.trim())
  if (parts.some((part) => part === '')) return null
  const last = Number(parts[parts.length - 1])
  if (!Number.isFinite(last)) return null
  if (parts.length === 3) {
    const hours = Number(parts[0])
    const mins = Number(parts[1])
    if (!Number.isFinite(hours) || !Number.isFinite(mins)) return null
    return hours * 3600 + mins * 60 + last
  }
  if (parts.length === 2) {
    const mins = Number(parts[0])
    if (!Number.isFinite(mins)) return null
    return mins * 60 + last
  }
  if (parts.length === 1) {
    return last
  }
  return null
}

const toFiniteNumber = (value) => {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

const getContainMetrics = (naturalWidth, naturalHeight, containerWidth, containerHeight) => {
  if (!naturalWidth || !naturalHeight || !containerWidth || !containerHeight) return null
  const scale = Math.min(containerWidth / naturalWidth, containerHeight / naturalHeight)
  const drawWidth = naturalWidth * scale
  const drawHeight = naturalHeight * scale
  const offsetX = (containerWidth - drawWidth) / 2
  const offsetY = (containerHeight - drawHeight) / 2
  return { scale, offsetX, offsetY, drawWidth, drawHeight }
}

const coerceBBox = (bbox) => {
  if (!bbox) return null
  if (Array.isArray(bbox)) return bbox
  if (typeof bbox === 'string') {
    const trimmed = bbox.trim()
    if (!trimmed) return null
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed) || (parsed && typeof parsed === 'object')) {
        return coerceBBox(parsed)
      }
    } catch (err) {
      const matches = trimmed.match(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi)
      if (matches && matches.length >= 4) {
        return matches.slice(0, 4).map((value) => Number(value))
      }
    }
    return null
  }
  if (typeof bbox === 'object') {
    if (
      'x1' in bbox &&
      'y1' in bbox &&
      'x2' in bbox &&
      'y2' in bbox
    ) {
      return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
    }
    if (
      'left' in bbox &&
      'top' in bbox &&
      'right' in bbox &&
      'bottom' in bbox
    ) {
      return [bbox.left, bbox.top, bbox.right, bbox.bottom]
    }
    if ('x' in bbox && 'y' in bbox && 'w' in bbox && 'h' in bbox) {
      const x = Number(bbox.x)
      const y = Number(bbox.y)
      const w = Number(bbox.w)
      const h = Number(bbox.h)
      return [x, y, x + w, y + h]
    }
  }
  return null
}

const normalizeBBox = (bbox, naturalWidth, naturalHeight) => {
  const raw = coerceBBox(bbox)
  if (!raw || raw.length < 4) return null
  let [x1, y1, x2, y2] = raw.slice(0, 4).map((value) => Number(value))
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null

  const hasSize = naturalWidth > 0 && naturalHeight > 0
  const maxVal = Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2))
  if (hasSize && maxVal <= 1.01) {
    x1 *= naturalWidth
    y1 *= naturalHeight
    x2 *= naturalWidth
    y2 *= naturalHeight
  }

  if (x2 < x1 || y2 < y1) {
    const maybeWidthHeight =
      hasSize &&
      x2 >= 0 &&
      y2 >= 0 &&
      x1 + x2 <= naturalWidth + 1 &&
      y1 + y2 <= naturalHeight + 1
    if (maybeWidthHeight) {
      x2 = x1 + x2
      y2 = y1 + y2
    }
  }

  if (x2 < x1) [x1, x2] = [x2, x1]
  if (y2 < y1) [y1, y2] = [y2, y1]
  return [x1, y1, x2, y2]
}

const getBBoxStyle = (bbox, naturalWidth, naturalHeight, containerWidth, containerHeight) => {
  const normalized = normalizeBBox(bbox, naturalWidth, naturalHeight)
  if (!normalized || normalized.length !== 4) return null
  const metrics = getContainMetrics(naturalWidth, naturalHeight, containerWidth, containerHeight)
  if (!metrics) return null
  const [x1, y1, x2, y2] = normalized.map((v) => Number(v))
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null
  const clampedX1 = clamp(x1, 0, naturalWidth)
  const clampedY1 = clamp(y1, 0, naturalHeight)
  const clampedX2 = clamp(x2, 0, naturalWidth)
  const clampedY2 = clamp(y2, 0, naturalHeight)
  if (clampedX2 <= clampedX1 || clampedY2 <= clampedY1) return null
  return {
    left: metrics.offsetX + clampedX1 * metrics.scale,
    top: metrics.offsetY + clampedY1 * metrics.scale,
    width: (clampedX2 - clampedX1) * metrics.scale,
    height: (clampedY2 - clampedY1) * metrics.scale,
  }
}

const useElementSize = (ref) => {
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    if (!ref.current) return undefined
    const element = ref.current
    const update = () => {
      setSize({
        width: element.clientWidth || 0,
        height: element.clientHeight || 0,
      })
    }
    update()
    if (typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])
  return size
}

const BBoxImage = ({ src, bbox, alt, className, imgClassName }) => {
  const containerRef = useRef(null)
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 })
  const { width: containerWidth, height: containerHeight } = useElementSize(containerRef)
  const boxStyle = useMemo(
    () =>
      getBBoxStyle(
        bbox,
        naturalSize.width,
        naturalSize.height,
        containerWidth,
        containerHeight
      ),
    [bbox, naturalSize, containerWidth, containerHeight]
  )

  return (
    <div ref={containerRef} className={`relative ${className || ''}`}>
      <img
        src={src}
        alt={alt}
        className={imgClassName}
        onLoad={(event) => {
          const img = event.currentTarget
          if (img?.naturalWidth && img?.naturalHeight) {
            setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight })
          }
        }}
      />
      {boxStyle && (
        <div
          className="pointer-events-none absolute rounded-sm border-2 border-emerald-400/90 shadow-[0_0_0_1px_rgba(5,150,105,0.6)]"
          style={boxStyle}
        />
      )}
    </div>
  )
}

const formatDurationMs = (value) => {
  const ms = toFiniteNumber(value)
  if (ms === null || ms < 0) return '--'
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`
  }
  return `${Math.round(ms)}ms`
}

const formatBytes = (value) => {
  const size = Number(value)
  if (!Number.isFinite(size) || size <= 0) return '--'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`
  return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

const formatClipLabel = (name, index = 0) => {
  if (!name) return `Location ${String(index + 1).padStart(2, '0')}`
  const base = String(name).replace(/\.[^/.]+$/, '')
  const match = base.match(/(\d+)$/)
  if (match) {
    return `Location ${String(Number(match[1])).padStart(2, '0')}`
  }
  return base.replace(/[_-]+/g, ' ')
}

const extractLocationNumber = (name) => {
  if (!name) return Number.POSITIVE_INFINITY
  const base = String(name).replace(/\.[^/.]+$/, '')
  const match = base.match(/(\d+)/)
  if (!match) return Number.POSITIVE_INFINITY
  const value = Number(match[1])
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY
}

const resolveClipPreview = (clip) => {
  if (!clip) return ''
  if (clip.thumbnail) return clip.thumbnail
  if (clip.thumbnail_url) return clip.thumbnail_url
  if (clip.preview_url) return clip.preview_url
  return ''
}

const toAbsoluteClipUrl = (streamUrl) => {
  if (!streamUrl) return ''
  if (streamUrl.startsWith('http://') || streamUrl.startsWith('https://')) return streamUrl
  if (streamUrl.startsWith('/')) return `${API_BASE_URL}${streamUrl}`
  return `${API_BASE_URL}/${streamUrl}`
}

const resolveThumbnail = (thumb) => {
  if (!thumb) return null
  if (thumb.startsWith('http') || thumb.startsWith('data:')) return thumb
  if (thumb.startsWith('/')) return `${API_BASE_URL}${thumb}`
  return `${API_BASE_URL}/${thumb}`
}

const normalizePlateList = (plates) => {
  if (!Array.isArray(plates)) return []
  return plates
    .map((plate) => (plate == null ? '' : String(plate).trim()))
    .filter(Boolean)
}

const extractTrackId = (item) => {
  const direct =
    item?.track_id ?? item?.trackId ?? item?.vehicle_id ?? item?.vehicleId ?? null
  if (Number.isFinite(Number(direct))) return Number(direct)
  const desc = String(item?.description || '')
  const match = desc.match(/(?:track|id)[:#\s]+(\d+)/i)
  if (match && Number.isFinite(Number(match[1]))) return Number(match[1])
  return null
}

const derivePlatesFromVehicles = (vehicles) => {
  if (!Array.isArray(vehicles)) return []
  const seen = new Set()
  vehicles.forEach((vehicle) => {
    const plate = vehicle?.plate_text ?? vehicle?.plate ?? vehicle?.plate_number ?? null
    if (!plate) return
    const value = String(plate).trim()
    if (!value || value.toLowerCase() === 'plate not visible') return
    seen.add(value)
  })
  return Array.from(seen)
}

const resolveSummaryImage = (image) => {
  if (!image) return null
  if (typeof image === 'string') return resolveThumbnail(image)
  if (typeof image === 'object') {
    const candidate = image.url || image.thumbnail || image.src || image.image || ''
    if (candidate) return resolveThumbnail(String(candidate))
  }
  return null
}

const resolveBBox = (item) =>
  item?.bbox ??
  item?.vehicle_bbox ??
  item?.best_bbox ??
  item?.bounding_box ??
  item?.box ??
  null

const normalizeResults = (items, report, offsetSec = 0) => {
  const sourceItems = items || []
  return sourceItems.map((item, index) => {
    const rawType = String(item.type || report?.tool_used || 'event').toLowerCase()
    const plateText =
      item.plate_text ||
      item.plate ||
      item.license_plate ||
      item.license_plate_text ||
      item.plate_number ||
      null
    const timeValue =
      typeof item.timestamp === 'string' && item.timestamp.includes(':')
        ? item.timestamp
        : item.time_sec ?? item.timestamp
    const timeSec = typeof timeValue === 'number' ? timeValue : parseTimestamp(timeValue)
    const absoluteTimeSec = timeSec != null ? timeSec + offsetSec : null
    const rawConf = item.confidence ?? item.plate_confidence ?? null
    const confidence =
      typeof rawConf === 'number'
        ? Math.round(rawConf <= 1 ? rawConf * 100 : rawConf)
        : item.severity === 'HIGH'
          ? 90
          : item.severity === 'MEDIUM'
            ? 70
            : item.severity === 'LOW'
              ? 50
              : null

    return {
      id: `${index}-${rawType}`,
      resultIndex: index,
      type: rawType,
      timeSec,
      absoluteTimeSec,
      frame: item.frame ?? null,
      timestamp: formatTimestamp(absoluteTimeSec ?? timeSec, { showMs: true }),
      segmentTimestamp: formatTimestamp(timeSec, { showMs: true }),
      confidence,
      severity: item.severity,
      thumbnail: resolveThumbnail(item.thumbnail),
      bbox: resolveBBox(item),
      verified: item.vlm_verified === true,
      verification: item.vlm_verified ?? null,
      vlmResponse: item.vlm_response ?? null,
      plateText: plateText ? String(plateText) : null,
      plateConfidence: typeof item.plate_confidence === 'number' ? item.plate_confidence : null,
      plateFormatValid: item.plate_format_valid ?? null,
      plateOriginal: item.plate_original ?? null,
      plateFormat: item.plate_format ?? null,
      vehicleType: item.vehicle_type ?? null,
      trackId: extractTrackId(item),
      description:
        item.description ||
        item.reason ||
        plateText ||
        item.object ||
        'Event detected',
    }
  })
}

const sortByFirstObserved = (items) => {
  const sorted = [...items]
  sorted.sort((a, b) => {
    const aTime = a.absoluteTimeSec ?? a.timeSec
    const bTime = b.absoluteTimeSec ?? b.timeSec
    const aVal = Number.isFinite(aTime) ? aTime : Number.POSITIVE_INFINITY
    const bVal = Number.isFinite(bTime) ? bTime : Number.POSITIVE_INFINITY
    if (aVal !== bVal) return aVal - bVal
    const aTrack = Number.isFinite(a.trackId) ? a.trackId : Number.POSITIVE_INFINITY
    const bTrack = Number.isFinite(b.trackId) ? b.trackId : Number.POSITIVE_INFINITY
    if (aTrack !== bTrack) return aTrack - bTrack
    return 0
  })
  return sorted
}

export function ForensicLab({ onTimingStats }) {
  const [query, setQuery] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [recordedClips, setRecordedClips] = useState([])
  const [clipsLoading, setClipsLoading] = useState(false)
  const [clipsError, setClipsError] = useState('')
  const [selectedClipId, setSelectedClipId] = useState('')
  const [referenceImageFile, setReferenceImageFile] = useState(null)
  const [referenceImageSrc, setReferenceImageSrc] = useState('')
  const [videoSrc, setVideoSrc] = useState('')
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState([])
  const [report, setReport] = useState(null)
  const [responseType, setResponseType] = useState('evidence')
  const [textAnswer, setTextAnswer] = useState('')
  const [plateList, setPlateList] = useState([])
  const [error, setError] = useState('')
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState('idle')
  const [isCancelling, setIsCancelling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [displayProgress, setDisplayProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState('')
  const [, setTimingStats] = useState(null)
  const [isTrimming, setIsTrimming] = useState(false)
  const [trimWarning, setTrimWarning] = useState('')
  const [isProbingDuration, setIsProbingDuration] = useState(false)
  const [segmentStart, setSegmentStart] = useState(0)
  const [segmentEnd, setSegmentEnd] = useState(0)
  const [selectedEvent, setSelectedEvent] = useState(null)
  const [, setIsSegmentDragging] = useState(false)
  const [dialogEvent, setDialogEvent] = useState(null)
  const [isEventDialogOpen, setIsEventDialogOpen] = useState(false)
  const [thumbnailLoadingId, setThumbnailLoadingId] = useState(null)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [needsServerPreview, setNeedsServerPreview] = useState(false)
  const fileInputRef = useRef(null)
  const imageInputRef = useRef(null)
  const pollRef = useRef(null)
  const pollFailuresRef = useRef(0)
  const pollInFlightRef = useRef(false)
  const submitAbortRef = useRef(null)
  const uploadXhrRef = useRef(null)
  const videoRef = useRef(null)
  const timelineRef = useRef(null)
  const segmentTrackRef = useRef(null)
  const wasPlayingRef = useRef(false)
  const ffmpegRef = useRef(null)
  const ffmpegLoadingRef = useRef(null)
  const analysisOffsetRef = useRef(0)
  const analysisStartRef = useRef(null)
  const uploadStartRef = useRef(null)
  const uploadDoneRef = useRef(null)
  const segmentDragHandleRef = useRef(null)
  const requestedThumbnailIdsRef = useRef(new Set())
  const selectedClip =
    recordedClips.find((clip) => String(clip.id) === String(selectedClipId)) || null
  const isFaceSearchMode = Boolean(referenceImageFile)
  const scanSteps = useMemo(
    () => (isFaceSearchMode ? MAGIC_SCAN_STEPS.face : MAGIC_SCAN_STEPS.general),
    [isFaceSearchMode]
  )

  const videoContainerRef = useRef(null)
  const { width: videoContainerWidth, height: videoContainerHeight } = useElementSize(videoContainerRef)
  const activeVideoBBox =
    selectedEvent?.bbox && selectedEvent?.absoluteTimeSec != null
      ? Math.abs((currentTime || 0) - selectedEvent.absoluteTimeSec) <= 0.75
        ? selectedEvent.bbox
        : null
      : selectedEvent?.bbox || null
  const videoBoxStyle = useMemo(() => {
    const video = videoRef.current
    if (!video || !activeVideoBBox) return null
    return getBBoxStyle(
      activeVideoBBox,
      video.videoWidth || 0,
      video.videoHeight || 0,
      videoContainerWidth,
      videoContainerHeight
    )
  }, [activeVideoBBox, videoContainerWidth, videoContainerHeight, currentTime])

  const resetAnalysisState = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    setIsAnalyzing(false)
    setResults([])
    setReport(null)
    setResponseType('evidence')
    setTextAnswer('')
    setPlateList([])
    setError('')
    setJobId(null)
    setJobStatus('idle')
    setIsCancelling(false)
    setProgress(0)
    setUploadProgress(0)
    setDisplayProgress(0)
    setStatusMessage('')
    setTimingStats(null)
    setIsTrimming(false)
    setTrimWarning('')
    setIsProbingDuration(false)
    analysisOffsetRef.current = 0
    setSegmentStart(0)
    setSegmentEnd(0)
    setSelectedEvent(null)
    setDialogEvent(null)
    setIsEventDialogOpen(false)
    setDuration(0)
    setCurrentTime(0)
    setIsPlaying(false)
    setNeedsServerPreview(false)
    setThumbnailLoadingId(null)
    requestedThumbnailIdsRef.current = new Set()
    analysisStartRef.current = null
    uploadStartRef.current = null
    uploadDoneRef.current = null
    submitAbortRef.current = null
    uploadXhrRef.current = null
  }

  const loadRecordedClips = useCallback(async () => {
    setClipsLoading(true)
    setClipsError('')
    try {
      const resp = await fetchWithRetry(
        `${API_BASE_URL}/recorded/clips`,
        {},
        { timeoutMs: DEFAULT_TIMEOUT_MS, retries: 1, retryDelayMs: 1000 }
      )
      const data = await readJsonResponse(resp)
      const clips = Array.isArray(data?.clips) ? data.clips : []
      const normalized = clips
        .map((clip) => {
          const streamUrl = toAbsoluteClipUrl(clip.stream_url || clip.streamUrl || '')
          const thumbnailUrl = streamUrl ? `${streamUrl}?thumb=1` : ''
          return {
            ...clip,
            stream_url: streamUrl,
            thumbnail_url: clip.thumbnail_url || thumbnailUrl,
            preview_url: clip.preview_url || streamUrl,
          }
        })
        .sort((a, b) => {
          const aOrder = extractLocationNumber(a.name)
          const bOrder = extractLocationNumber(b.name)
          if (aOrder !== bOrder) return aOrder - bOrder
          return String(a.name || '').localeCompare(String(b.name || ''))
        })
      setRecordedClips(normalized)
    } catch (err) {
      setRecordedClips([])
      setClipsError(err?.message || 'Unable to load recorded clips.')
    } finally {
      setClipsLoading(false)
    }
  }, [])

  const handleSelectRecordedClip = (clip) => {
    if (!clip) return
    resetAnalysisState()
    setSelectedClipId(clip.id)
    setSelectedFile(null)
    setPreviewError('')
    setNeedsServerPreview(false)
    if (videoSrc && videoSrc.startsWith('blob:')) {
      URL.revokeObjectURL(videoSrc)
    }
    const streamUrl = toAbsoluteClipUrl(clip.stream_url || clip.streamUrl || clip.preview_url || '')
    setVideoSrc(streamUrl)
  }

  const handlePickImage = () => {
    imageInputRef.current?.click()
  }

  const clearReferenceImage = () => {
    setReferenceImageFile(null)
    setReferenceImageSrc('')
    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }

  const handleImageChange = (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setReferenceImageFile(file)
    const reader = new FileReader()
    reader.onload = () => {
      setReferenceImageSrc(String(reader.result || ''))
    }
    reader.onerror = () => {
      setReferenceImageSrc('')
    }
    reader.readAsDataURL(file)
    event.target.value = ''
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    resetAnalysisState()
    setSelectedFile(file)
    setSelectedClipId('')
    setPreviewError('')
    setIsPreviewLoading(false)
    if (videoSrc && videoSrc.startsWith('blob:')) {
      URL.revokeObjectURL(videoSrc)
    }
    const canPlay = (() => {
      if (typeof document === 'undefined' || !file.type) return true
      const video = document.createElement('video')
      const result = video.canPlayType(file.type)
      return result === 'probably' || result === 'maybe'
    })()
    if (canPlay) {
      setVideoSrc(URL.createObjectURL(file))
      setNeedsServerPreview(false)
      setIsProbingDuration(true)
    } else {
      setVideoSrc('')
      setNeedsServerPreview(true)
      setIsProbingDuration(false)
      setDuration(0)
      setSegmentStart(0)
      setSegmentEnd(0)
    }
    event.target.value = ''
  }

  const handleTimeUpdate = () => {
    if (isScrubbing) return
    const video = videoRef.current
    if (!video) return
    setCurrentTime(video.currentTime || 0)
  }

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    if (!video) return
    const nextDuration = Number.isFinite(video.duration) ? video.duration : 0
    setDuration(nextDuration)
    setCurrentTime(video.currentTime || 0)
    setIsProbingDuration(false)
    if (nextDuration > 0) {
      setSegmentStart(0)
      setSegmentEnd(nextDuration)
    }
  }

  const handleTogglePlay = () => {
    if (!videoSrc || needsServerPreview) return
    const video = videoRef.current
    if (!video) return
    if (video.paused || video.ended) {
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }

  const seekToClientX = (clientX) => {
    const track = timelineRef.current
    if (!track || !duration) return
    const rect = track.getBoundingClientRect()
    if (!rect.width) return
    const pct = clamp((clientX - rect.left) / rect.width, 0, 1)
    const time = pct * duration
    const video = videoRef.current
    if (video) {
      video.currentTime = time
    }
    setCurrentTime(time)
  }

  const handleSeek = (event) => {
    if (!duration) return
    seekToClientX(event.clientX)
  }

  const handleScrubStart = (event) => {
    if (!duration) return
    event.preventDefault()
    setIsScrubbing(true)
    wasPlayingRef.current = isPlaying
    if (videoRef.current) {
      videoRef.current.pause()
    }
    seekToClientX(event.clientX)
    const handleMove = (moveEvent) => {
      seekToClientX(moveEvent.clientX)
    }
    const handleUp = () => {
      setIsScrubbing(false)
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
      if (wasPlayingRef.current && videoRef.current) {
        videoRef.current.play().catch(() => {})
      }
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp, { once: true })
  }

  const handleSegmentDragStart = (edge) => (event) => {
    if (!duration) return
    event.preventDefault()
    const track = segmentTrackRef.current
    if (!track) return
    setIsSegmentDragging(true)
    segmentDragHandleRef.current = edge
    const rect = track.getBoundingClientRect()
    const lockedStart = segmentStart
    const lockedEnd = segmentEnd

    const updateFromClientX = (clientX) => {
      if (!rect.width) return
      const pct = clamp((clientX - rect.left) / rect.width, 0, 1)
      const rawTime = pct * duration
      const stepped = Math.round(rawTime / SEGMENT_STEP) * SEGMENT_STEP
      if (edge === 'start') {
        const maxStart = Math.max(0, lockedEnd - SEGMENT_MIN_GAP)
        setSegmentStart(clamp(stepped, 0, maxStart))
      } else {
        const minEnd = Math.min(duration, lockedStart + SEGMENT_MIN_GAP)
        setSegmentEnd(clamp(stepped, minEnd, duration))
      }
    }

    updateFromClientX(event.clientX)

    const handleMove = (moveEvent) => updateFromClientX(moveEvent.clientX)
    const handleUp = () => {
      setIsSegmentDragging(false)
      segmentDragHandleRef.current = null
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp, { once: true })
  }

  const handleSelectEvent = (eventItem) => {
    if (!eventItem) return
    setSelectedEvent(eventItem)
    const targetTime = toFiniteNumber(eventItem.absoluteTimeSec ?? eventItem.timeSec)
    if (targetTime != null) {
      if (videoRef.current) {
        videoRef.current.currentTime = targetTime
      }
      setCurrentTime(targetTime)
    }
  }

  const handleOpenEventDialog = (eventItem) => {
    if (!eventItem) return
    setDialogEvent(eventItem)
    setIsEventDialogOpen(true)
  }

  const ensureFFmpeg = async () => {
    if (ffmpegRef.current) return ffmpegRef.current
    if (ffmpegLoadingRef.current) {
      await ffmpegLoadingRef.current
      return ffmpegRef.current
    }
    const ffmpeg = new FFmpeg()
    ffmpegRef.current = ffmpeg
    ffmpegLoadingRef.current = ffmpeg.load({ coreURL, wasmURL })
    await ffmpegLoadingRef.current
    return ffmpeg
  }

  const trimVideoSegment = async (file, startSec, endSec) => {
    if (!file) throw new Error('No video selected for trimming.')
    const safeStart = Math.max(0, Number(startSec) || 0)
    const safeEnd = Math.max(safeStart, Number(endSec) || 0)
    if (safeEnd <= safeStart) throw new Error('Invalid trim range.')
    const ffmpeg = await ensureFFmpeg()
    const sanitizedName = String(file.name || 'upload')
      .replace(/[^\w.-]+/g, '_')
      .slice(0, 80)
    const inputName = `input_${Date.now()}_${sanitizedName}`
    const outputName = `output_${Date.now()}.mp4`

    await ffmpeg.writeFile(inputName, await fetchFile(file))

    const startArg = safeStart.toFixed(3)
    const durationArg = Math.max(0, safeEnd - safeStart).toFixed(3)
    let usedFastCopy = false

    try {
      if (!FAST_TRIM_MODE) {
        throw new Error('Fast trim disabled.')
      }
      await ffmpeg.exec(['-ss', startArg, '-t', durationArg, '-i', inputName, '-c', 'copy', outputName])
      usedFastCopy = true
    } catch (err) {
      usedFastCopy = false
      await ffmpeg.exec([
        '-ss',
        startArg,
        '-t',
        durationArg,
        '-i',
        inputName,
        '-c:v',
        'libx264',
        '-c:a',
        'aac',
        '-movflags',
        'faststart',
        outputName,
      ])
    }

    const data = await ffmpeg.readFile(outputName)
    await ffmpeg.deleteFile(inputName)
    await ffmpeg.deleteFile(outputName)

    const trimmedFile = new File([data.buffer], `${sanitizedName}_trim.mp4`, {
      type: 'video/mp4',
    })
    return { file: trimmedFile, usedFastCopy }
  }

  useEffect(() => {
    loadRecordedClips()
  }, [loadRecordedClips])

  const startJobPolling = (newJobId) => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
    }
    pollFailuresRef.current = 0

    pollRef.current = setInterval(async () => {
      if (pollInFlightRef.current) return
      pollInFlightRef.current = true
      try {
        const statusResp = await fetchWithTimeout(
          `${API_BASE_URL}/jobs/${newJobId}`,
          {},
          STATUS_TIMEOUT_MS
        )
        const statusData = await readJsonResponse(statusResp)
        pollFailuresRef.current = 0
        if (error) setError('')
        const uploadPct =
          typeof statusData.upload_progress === 'number'
            ? statusData.upload_progress
            : uploadProgress
        const normalizedStatus = String(statusData.status || '').trim().toLowerCase()

        setJobStatus(statusData.status || 'running')
        setUploadProgress(uploadPct)
        setProgress(Math.max(Number(statusData.progress || 0), uploadPct || 0))
        setStatusMessage(statusData.message || '')
        setIsCancelling(Boolean(statusData.cancel_requested))

        if (normalizedStatus === 'completed') {
          const answerReceivedAt = performance.now()
          const backendTimings = statusData.timings || statusData.result?.timings || {}
          const clientUploadMs =
            uploadStartRef.current != null && uploadDoneRef.current != null
              ? uploadDoneRef.current - uploadStartRef.current
              : null
          const clientPostUploadToAnswerMs =
            uploadDoneRef.current != null ? answerReceivedAt - uploadDoneRef.current : null
          const clientEndToEndMs =
            analysisStartRef.current != null ? answerReceivedAt - analysisStartRef.current : null

          const nextTimingStats = {
            client_upload_ms: clientUploadMs,
            client_post_upload_to_answer_ms: clientPostUploadToAnswerMs,
            client_end_to_end_ms: clientEndToEndMs,
            backend_request_to_upload_ms: toFiniteNumber(backendTimings.request_to_upload_ms),
            backend_queue_wait_ms: toFiniteNumber(backendTimings.queue_wait_ms),
            backend_processing_ms: toFiniteNumber(backendTimings.processing_ms),
            backend_post_upload_to_complete_ms: toFiniteNumber(
              backendTimings.post_upload_to_complete_ms
            ),
            backend_end_to_end_request_ms: toFiniteNumber(
              backendTimings.end_to_end_request_ms
            ),
          }
          setTimingStats(nextTimingStats)
          if (onTimingStats) {
            onTimingStats(nextTimingStats)
          }

          setReport(statusData.result)
          
          const mode = statusData.result?.response_type || 'evidence'
          setResponseType(mode)
          setTextAnswer(statusData.result?.text_answer || '')

          const toolUsed = statusData.result?.tool_used || null
          const intentConfig = statusData.result?.intent_config || null

          let rawResults = tryParseJson(statusData.result?.results)
          if (!Array.isArray(rawResults) && typeof rawResults === 'object' && rawResults?.events) {
            rawResults = rawResults.events
          }
          if (!Array.isArray(rawResults)) rawResults = tryParseJson(statusData.result?.events)
          if (!Array.isArray(rawResults)) rawResults = []

          // Extract plates from results if LPR tool was used
          const plates = []
          if (toolUsed === 'LPR' && Array.isArray(rawResults)) {
            rawResults.forEach(item => {
              const plate = item?.plate_text || item?.plate || item?.license_plate || null
              if (plate && typeof plate === 'string' && plate.trim()) {
                plates.push(plate.trim())
              }
            })
          }
          setPlateList(plates)

          const normalized =
            mode === 'text'
              ? []
              : normalizeResults(
                  Array.isArray(rawResults) ? rawResults : [],
                  statusData.result,
                  analysisOffsetRef.current
                )
          const sortedResults = sortByFirstObserved(normalized)

          setResults(sortedResults)
          setSelectedEvent(sortedResults[0] || null)
          setProgress(100)
          setIsAnalyzing(false)
          setIsCancelling(false)
          clearInterval(pollRef.current)
          pollRef.current = null
        }

        if (normalizedStatus === 'failed') {
          setError(statusData.error || 'Analysis failed.')
          setIsAnalyzing(false)
          setIsCancelling(false)
          clearInterval(pollRef.current)
          pollRef.current = null
        }

        if (normalizedStatus === 'canceled') {
          setIsAnalyzing(false)
          setIsCancelling(false)
          setJobStatus('canceled')
          setStatusMessage(statusData.message || 'Analysis canceled.')
          clearInterval(pollRef.current)
          pollRef.current = null
        }

        if (
          normalizedStatus !== 'completed' &&
          statusData.partial_results &&
          statusData.partial_results.length > 0
        ) {
          const mode = statusData.response_type || 'evidence'
          if (mode !== 'text') {
            const reportLike = { tool_used: statusData.tool_used }
            const partial = normalizeResults(
              statusData.partial_results,
              reportLike,
              analysisOffsetRef.current
            )
            const sortedPartial = sortByFirstObserved(partial)
            setResults(sortedPartial)
            if (!selectedEvent) {
              setSelectedEvent(sortedPartial[0] || null)
            }
          }
        }
      } catch (pollErr) {
        pollFailuresRef.current += 1
        if (pollFailuresRef.current >= 3) {
          setError(pollErr?.message || 'Failed to fetch job status.')
          setStatusMessage('Connection unstable. Stopping...')
          clearInterval(pollRef.current)
          pollRef.current = null
          setIsAnalyzing(false)
          setIsCancelling(false)
        } else {
          setStatusMessage('Reconnecting to server...')
        }
      } finally {
        pollInFlightRef.current = false
      }
    }, 1000)
  }

  const uploadSingle = async (file, queryText, referenceImageFile = null, clipId = null) => {
    const hasReferenceImage = Boolean(referenceImageFile)
    const formData = new FormData()
    if (file) {
      formData.append('video', file)
    } else if (clipId) {
      formData.append('clip_id', clipId)
    } else {
      throw new Error('Video source is required.')
    }
    formData.append('query', queryText || '')
    formData.append('include_thumbnails', 'false')
    if (hasReferenceImage) {
      formData.append('reference_image', referenceImageFile)
    }
    uploadStartRef.current = performance.now()
    uploadDoneRef.current = null

    let attempt = 0
    let data = null

    while (true) {
      try {
        const endpoint = clipId ? '/analyze_recorded' : '/analyze'
        const text = await uploadWithProgress(`${API_BASE_URL}${endpoint}`, formData, {
          timeoutMs: UPLOAD_TIMEOUT_MS,
          onStart: (xhr) => {
            uploadXhrRef.current = xhr
          },
          onProgress: (pct) => {
            setUploadProgress(pct)
            setProgress((prev) => Math.max(prev, pct))
            setStatusMessage(
              `${hasReferenceImage ? 'Uploading with reference image' : 'Uploading'} ${pct}%`
            )
          },
        })
        try {
          data = text ? JSON.parse(text) : null
        } catch (parseErr) {
          throw new Error('Invalid JSON response from server.')
        }
        uploadDoneRef.current = performance.now()
        break
      } catch (err) {
        if (attempt >= 2 || !isRetryableError(err)) {
          throw err
        }
        attempt += 1
        setStatusMessage(
          `${hasReferenceImage ? 'Retrying reference upload' : 'Retrying upload'} (attempt ${attempt})...`
        )
        await sleep(1000 * Math.pow(2, attempt - 1))
      } finally {
        uploadXhrRef.current = null
      }
    }

    if (!data?.job_id) {
      throw new Error('Upload failed to return job id.')
    }

    setJobId(data.job_id)
    setJobStatus('queued')
    setStatusMessage('Queued')
    setUploadProgress(100)
    startJobPolling(data.job_id)

    if (needsServerPreview) {
      setIsPreviewLoading(true)
      setPreviewError('')
      try {
        const previewResp = await fetchWithRetry(
          `${API_BASE_URL}/preview/${data.job_id}`,
          {},
          {
            timeoutMs: PREVIEW_TIMEOUT_MS,
            retries: 1,
            retryDelayMs: 1000,
          }
        )
        if (!previewResp.ok) {
          const message = await previewResp.text()
          throw new Error(message || 'Preview conversion failed.')
        }
        const blob = await previewResp.blob()
        if (videoSrc && videoSrc.startsWith('blob:')) {
          URL.revokeObjectURL(videoSrc)
        }
        setVideoSrc(URL.createObjectURL(blob))
        setNeedsServerPreview(false)
      } catch (err) {
        setPreviewError(err?.message || 'Unable to generate preview.')
      } finally {
        setIsPreviewLoading(false)
      }
    }
  }

  const analyzeRecordedSingle = async (clipId, queryText, referenceImageFile = null) => {
    const formData = new FormData()
    formData.append('clip_id', clipId)
    formData.append('query', queryText || '')
    formData.append('include_thumbnails', 'false')
    if (referenceImageFile) {
      formData.append('reference_image', referenceImageFile)
    }

    const controller = new AbortController()
    submitAbortRef.current = controller
    try {
      const resp = await fetchWithRetry(
        `${API_BASE_URL}/analyze_recorded`,
        { method: 'POST', body: formData, signal: controller.signal },
        {
          timeoutMs: UPLOAD_TIMEOUT_MS,
          retries: 2,
          retryDelayMs: 1000,
          onRetry: (attempt) => {
            setStatusMessage(`Retrying recorded request (attempt ${attempt})...`)
          },
        },
      )
      const data = await readJsonResponse(resp)
      if (!data?.job_id) {
        throw new Error('Recorded analysis failed to return job id.')
      }
      setJobId(data.job_id)
      setJobStatus('queued')
      setUploadProgress(100)
      setProgress(5)
      setStatusMessage('Queued')
      startJobPolling(data.job_id)
    } finally {
      submitAbortRef.current = null
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile && !selectedClip) {
      setError('Please select a recorded clip or upload a video first.')
      return
    }
    if (!query.trim()) {
      setError('Please enter a query to analyze.')
      return
    }

    setIsAnalyzing(true)
    setIsTrimming(false)
    setTrimWarning('')
    setError('')
    setPlateList([])
    setJobStatus('queued')
    setIsCancelling(false)
    setProgress(0)
    setUploadProgress(0)
    setStatusMessage('Preparing upload')
    setTimingStats(null)
    analysisStartRef.current = performance.now()
    uploadStartRef.current = null
    uploadDoneRef.current = null

    try {
      if (!selectedFile && selectedClip) {
        analysisOffsetRef.current = 0
        setStatusMessage(
          referenceImageFile
            ? 'Submitting recorded clip with reference image'
            : 'Submitting recorded clip'
        )
        await analyzeRecordedSingle(String(selectedClip.id), query.trim(), referenceImageFile)
        return
      }

      let fileToUpload = selectedFile
      let offsetForAnalysis = 0
      const hasDuration = duration > 0
      const fullSelection =
        !hasDuration ||
        (segmentStart <= SEGMENT_EPSILON && segmentEnd >= duration - SEGMENT_EPSILON)

      if (hasDuration && !fullSelection && segmentEnd > segmentStart + SEGMENT_MIN_GAP) {
        setIsTrimming(true)
        setStatusMessage('Trimming selected segment locally')
        try {
          const trimResult = await trimVideoSegment(selectedFile, segmentStart, segmentEnd)
          fileToUpload = trimResult.file
          if (trimResult.usedFastCopy) {
            setTrimWarning(
              'Fast trim enabled. Segment start may snap to the nearest keyframe.'
            )
          }
          offsetForAnalysis = segmentStart
        } catch (trimErr) {
          setTrimWarning(
            'Client-side trimming failed. Uploading the full video as fallback.'
          )
          fileToUpload = selectedFile
          offsetForAnalysis = 0
        } finally {
          setIsTrimming(false)
        }
      }

      analysisOffsetRef.current = offsetForAnalysis
      setStatusMessage('Uploading')
      await uploadSingle(fileToUpload, query.trim(), referenceImageFile)
    } catch (err) {
      const message = err?.message || 'Failed to analyze the video.'
      setResults([])
      setReport(null)
      if (message.toLowerCase().includes('canceled')) {
        setError('')
        setJobStatus('canceled')
        setStatusMessage(message)
      } else {
        setError(message)
      }
      setIsAnalyzing(false)
      setIsCancelling(false)
    } finally {
      // Handled by polling
    }
  }

  const handleStopAnalysis = async () => {
    if (!isAnalyzing && !isCancelling) return

    if (submitAbortRef.current) {
      setIsCancelling(true)
      setStatusMessage('Stopping request...')
      submitAbortRef.current.abort()
      submitAbortRef.current = null
      setIsAnalyzing(false)
      setIsCancelling(false)
      setJobStatus('canceled')
      setStatusMessage('Request canceled.')
      return
    }

    if (uploadXhrRef.current) {
      setIsCancelling(true)
      setStatusMessage('Stopping upload...')
      uploadXhrRef.current.abort()
      uploadXhrRef.current = null
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
      setIsAnalyzing(false)
      setIsCancelling(false)
      setJobStatus('canceled')
      setStatusMessage('Upload canceled.')
      return
    }

    if (!jobId) {
      setIsCancelling(false)
      return
    }

    setIsCancelling(true)
    setStatusMessage('Stopping analysis...')
    setError('')
    try {
      const resp = await fetchWithRetry(
        `${API_BASE_URL}/jobs/${jobId}/cancel`,
        { method: 'POST' },
        { timeoutMs: DEFAULT_TIMEOUT_MS, retries: 1, retryDelayMs: 750 }
      )
      const data = await readJsonResponse(resp)
      if (String(data?.status || '').toLowerCase() === 'canceled') {
        if (pollRef.current) {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
        setIsAnalyzing(false)
        setIsCancelling(false)
        setJobStatus('canceled')
        setStatusMessage(data?.message || 'Analysis canceled.')
        return
      }
      setStatusMessage(data?.message || 'Stopping analysis...')
    } catch (err) {
      setIsCancelling(false)
      setError(err?.message || 'Failed to stop analysis.')
    }
  }

  const handleExport = () => {
    if (!report) return
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'forensic_report.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  const getTypeIcon = (type) => {
    const label = String(type || '').toLowerCase()
    if (label.includes('weapon') || label.includes('gun') || label.includes('knife')) {
      return <Crosshair className="h-4 w-4 text-rose-600" />
    }
    if (label.includes('crowd') || label.includes('people')) {
      return <Users className="h-4 w-4 text-amber-600" />
    }
    if (label.includes('vehicle') || label.includes('plate') || label.includes('lpr')) {
      return <Car className="h-4 w-4 text-sky-600" />
    }
    if (label.includes('violence') || label.includes('fight')) {
      return <AlertTriangle className="h-4 w-4 text-destructive" />
    }
    return <AlertTriangle className="h-4 w-4 text-muted-foreground" />
  }

  const getTypePillClass = (type) => {
    const label = String(type || '').toLowerCase()
    if (label.includes('weapon') || label.includes('gun') || label.includes('knife')) {
      return 'border-rose-500/30 bg-rose-500/10 text-rose-200'
    }
    if (label.includes('crowd') || label.includes('people')) {
      return 'border-amber-500/30 bg-amber-500/10 text-amber-200'
    }
    if (label.includes('vehicle') || label.includes('plate') || label.includes('lpr')) {
      return 'border-sky-500/30 bg-sky-500/10 text-sky-200'
    }
    if (label.includes('violence') || label.includes('fight')) {
      return 'border-red-500/30 bg-red-500/10 text-red-200'
    }
    return 'border-white/10 bg-white/5 text-muted-foreground'
  }

  const getTypeColor = (type) => {
    const label = String(type || '').toLowerCase()
    if (label.includes('weapon') || label.includes('gun') || label.includes('knife')) {
      return 'border-rose-500/30 bg-rose-500/10'
    }
    if (label.includes('crowd') || label.includes('people')) {
      return 'border-amber-500/30 bg-amber-500/10'
    }
    if (label.includes('vehicle') || label.includes('plate') || label.includes('lpr')) {
      return 'border-sky-500/30 bg-sky-500/10'
    }
    if (label.includes('violence') || label.includes('fight')) {
      return 'border-destructive/30 bg-destructive/10'
    }
    return 'border-border bg-muted/40'
  }

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0
  const segmentStartPct = duration > 0 ? (segmentStart / duration) * 100 : 0
  const segmentEndPct = duration > 0 ? (segmentEnd / duration) * 100 : 100
  const segmentDuration = Math.max(0, segmentEnd - segmentStart)
  const isFullSegment =
    !duration ||
    (segmentStart <= SEGMENT_EPSILON && segmentEnd >= duration - SEGMENT_EPSILON)
  const showVideoPreview = Boolean(videoSrc) && !needsServerPreview
  const hasSelectedSource = Boolean(selectedFile || selectedClip)
  const analysisProgressRaw = Math.min(100, Math.max(progress, displayProgress))
  const analysisProgressCap =
    isAnalyzing && jobStatus !== 'completed' ? 98 : 100
  const analysisProgress = Math.min(analysisProgressCap, analysisProgressRaw)
  const analysisProgressLabel = Math.round(analysisProgress)
  const progressGlow = Math.min(96, Math.max(2, analysisProgress))
  const canStopAnalysis = isAnalyzing || isCancelling
  const activeScanStep = useMemo(() => {
    if (!scanSteps.length) return MAGIC_SCAN_STEPS.general[0]
    const pct = Math.max(0, Math.min(100, analysisProgress))
    let current = scanSteps[0]
    scanSteps.forEach((step) => {
      if (pct >= step.at) current = step
    })
    return current || MAGIC_SCAN_STEPS.general[0]
  }, [scanSteps, analysisProgress])
  const primaryStatus = statusMessage || activeScanStep?.title || 'Analyzing'
  const secondaryStatus = activeScanStep?.detail || ''
  const markers = useMemo(() => {
    if (!duration) return []
    return results
      .map((result) => ({
        id: result.id,
        left: Math.min(
          100,
          Math.max(0, (((result.absoluteTimeSec ?? result.timeSec) || 0) / duration) * 100)
        ),
        highlight:
          result.type === 'weapon' || result.type === 'violence' ? 'bg-rose-500' : 'bg-amber-500',
      }))
      .filter((marker) => marker.left >= 0)
  }, [duration, results])

  const isTextMode = responseType === 'text'
  const hasResults = results.length > 0 || plateList.length > 0
  const showSegmentTimestamp =
    dialogEvent?.segmentTimestamp && dialogEvent.segmentTimestamp !== dialogEvent.timestamp
  const panelClass =
    'rounded-2xl border border-white/10 bg-black/60 p-4 backdrop-blur-sm shadow-[0_0_40px_rgba(15,23,42,0.45)]'
  const panelSoftClass =
    'rounded-2xl border border-white/10 bg-black/45 p-4 backdrop-blur-sm'
  const panelInsetClass = 'rounded-lg border border-white/10 bg-black/60'

  return (
    <div className="flex h-full flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-foreground">
              Forensic Review
            </h1>
            <p className="text-xs text-muted-foreground">
              Command center for CCTV triage and evidence extraction.
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs font-medium text-accent shadow-[0_0_18px_rgba(56,189,248,0.2)]">
            <Search className="h-3 w-3 text-accent" />
            Analysis
          </div>
        </div>
        {report && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            className="border-white/10 bg-white/5 hover:bg-white/10"
          >
            <Download className="mr-2 h-4 w-4" />
            Export Report
          </Button>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[2fr_1fr]">
        <div className="flex flex-col gap-4">
          <div className={`${panelClass}`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4 text-accent" />
                <p className="text-sm font-semibold text-foreground">Command Console</p>
              </div>
            </div>

            <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,520px)_minmax(0,360px)] xl:items-start">
              <div className="space-y-2">
                <div className="relative w-full">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    onKeyDown={(event) => event.key === 'Enter' && handleAnalyze()}
                    placeholder='Search: "Find fighting", "Show weapons", "person in black jacket"...'
                    className="border-white/10 bg-black/70 pl-10 text-foreground placeholder:text-muted-foreground/70 focus-visible:ring-2 focus-visible:ring-accent/60"
                  />
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button onClick={handleAnalyze} disabled={!hasSelectedSource || isAnalyzing}>
                    {isFaceSearchMode ? 'Face Search' : 'Analyze'}
                  </Button>
                  {canStopAnalysis && (
                    <Button
                      variant="outline"
                      onClick={handleStopAnalysis}
                      disabled={isCancelling}
                      className="border-red-500/30 bg-red-500/10 text-red-200 hover:bg-red-500/20"
                    >
                      <X className="mr-2 h-4 w-4" />
                      {isCancelling ? 'Stopping...' : 'Stop'}
                    </Button>
                  )}
                  <Button
                    variant={isFaceSearchMode ? 'default' : 'outline'}
                    onClick={handlePickImage}
                    disabled={isAnalyzing}
                  >
                    <ImageIcon className="mr-2 h-4 w-4" />
                    {isFaceSearchMode ? 'Change Face Image' : 'Add Face Image'}
                  </Button>
                  {selectedClip && (
                    <div className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent">
                      <FileVideo className="h-3.5 w-3.5" />
                      {formatClipLabel(selectedClip.name)}
                    </div>
                  )}
                  {selectedFile && (
                    <div className="inline-flex max-w-[240px] items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-foreground/80">
                      <Upload className="h-3.5 w-3.5" />
                      <span className="truncate">{selectedFile.name}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                {referenceImageFile && (
                  <div className="w-full max-w-[360px] rounded-xl border border-white/10 bg-black/45 p-2">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        Reference Image
                      </p>
                      <button
                        type="button"
                        onClick={clearReferenceImage}
                        className="rounded p-1 text-muted-foreground transition hover:bg-white/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
                        aria-label="Remove reference image"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    <div className="mt-2 aspect-[4/3] w-full max-w-[320px] overflow-hidden rounded-lg border border-white/10 bg-black/60">
                      {referenceImageSrc ? (
                        <img
                          src={referenceImageSrc}
                          alt="Reference"
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
                          No reference image
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {isAnalyzing && (
              <div className="mt-3 rounded-xl border border-white/10 bg-gradient-to-br from-slate-950/80 via-black/60 to-black/90 px-4 py-3 shadow-[0_0_30px_rgba(56,189,248,0.16)]">
                <div className="flex items-start justify-between gap-4 text-xs">
                  <div className="flex flex-col">
                    <span className="text-foreground/90">{primaryStatus}</span>
                    <span className="mt-1 text-[11px] text-muted-foreground/80">
                      {secondaryStatus}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 font-mono text-foreground/80">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/60 opacity-70" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
                    </span>
                    <span className="tabular-nums">{analysisProgressLabel}%</span>
                  </div>
                </div>
                <div className="relative mt-3 h-2 overflow-hidden rounded-full bg-muted/60">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_50%,rgba(56,189,248,0.2),transparent_60%)]" />
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-accent via-sky-400 to-emerald-400 shadow-[0_0_14px_rgba(56,189,248,0.45)] transition-all duration-500"
                    style={{ width: `${analysisProgress}%` }}
                  />
                  <div
                    className="absolute top-1/2 h-5 w-5 -translate-y-1/2 rounded-full bg-accent/30 blur-md"
                    style={{ left: `calc(${progressGlow}% - 10px)` }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent opacity-40 animate-pulse" />
                </div>
              </div>
            )}
          </div>

          <div className={`${panelClass}`}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-accent" />
                <p className="text-sm font-semibold text-foreground">Recorded CCTV Clips</p>
              </div>
              <Button variant="outline" size="sm" onClick={loadRecordedClips} disabled={clipsLoading}>
                <RefreshCw className={`mr-2 h-3.5 w-3.5 ${clipsLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>

            {clipsError && (
              <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {clipsError}
              </div>
            )}

            {clipsLoading && (
              <div className="mt-3 rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-muted-foreground">
                Loading recorded clips...
              </div>
            )}

            {!clipsLoading && recordedClips.length === 0 && !clipsError && (
              <div className="mt-3 rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-muted-foreground">
                No recorded clips found in the backend clips directory.
              </div>
            )}

            {recordedClips.length > 0 && (
              <div className="mt-3 max-h-[360px] overflow-y-auto pr-1">
                <div className="grid auto-rows-fr gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
                  {recordedClips.map((clip, index) => {
                    const isSelected = String(clip.id) === String(selectedClipId)
                    const preview = resolveClipPreview(clip)
                    return (
                      <button
                        key={clip.id}
                        type="button"
                        onClick={() => handleSelectRecordedClip(clip)}
                        className={`cursor-pointer rounded-xl border px-3 py-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-black ${isSelected
                          ? 'border-accent/50 bg-accent/10 shadow-[0_0_20px_rgba(56,189,248,0.12)]'
                          : 'border-white/10 bg-black/30 hover:border-accent/30 hover:bg-black/45'
                          }`}
                      >
                        <div className="relative mb-2 aspect-video w-full overflow-hidden rounded-lg border border-white/10 bg-black/60">
                          {preview ? (
                            <img
                              src={preview}
                              alt={`${clip.name} preview`}
                              className="h-full w-full object-cover"
                              loading="lazy"
                            />
                          ) : (
                            <div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
                              No preview
                            </div>
                          )}
                          <div className="absolute right-2 top-2 rounded-md border border-white/10 bg-black/60 px-1.5 py-1">
                            <Play className="h-3 w-3 text-foreground/70" />
                          </div>
                        </div>
                        <p className="text-sm font-medium text-foreground">
                          {formatClipLabel(clip.name, index)}
                        </p>
                        <p className="mt-1 truncate text-xs text-muted-foreground">{clip.name}</p>
                        <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
                          <span>{formatBytes(clip.size_bytes)}</span>
                          <span>
                            {clip.updated_at
                              ? new Date(Number(clip.updated_at) * 1000).toLocaleDateString()
                              : '--'}
                          </span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
          {!hasSelectedSource ? (
            <div className={`${panelSoftClass} flex flex-col items-center justify-center border-dashed px-6 py-12 text-center`}>
              <FileVideo className="mb-3 h-12 w-12 text-muted-foreground" />
              <p className="font-medium text-foreground">Select a recorded clip to begin</p>
              <p className="mt-2 text-xs text-muted-foreground">
                Uploads are disabled in recorded mode.
              </p>
            </div>
          ) : (
            <div className="relative w-full overflow-hidden rounded-2xl border border-border bg-black/95 shadow-[0_0_0_1px_rgba(255,255,255,0.04)]">
              <div
                ref={videoContainerRef}
                className="relative aspect-video w-full max-h-[360px] min-h-[220px]"
              >
                {videoSrc ? (
                  <>
                    <video
                      ref={videoRef}
                      src={videoSrc}
                      className={`absolute inset-0 h-full w-full object-contain ${needsServerPreview ? 'opacity-0 pointer-events-none' : ''
                        }`}
                      onTimeUpdate={handleTimeUpdate}
                      onLoadedMetadata={handleLoadedMetadata}
                      onClick={handleTogglePlay}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                      onEnded={() => setIsPlaying(false)}
                      playsInline
                      preload="metadata"
                    />
                    {videoBoxStyle && (
                      <div
                        className="pointer-events-none absolute rounded-sm border-2 border-emerald-400/90 shadow-[0_0_0_1px_rgba(5,150,105,0.6)]"
                        style={videoBoxStyle}
                      />
                    )}
                    {needsServerPreview && (
                      <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-muted/20 via-black/60 to-black">
                        <div className="text-center">
                          <FileVideo className="mx-auto mb-2 h-12 w-12 text-muted-foreground" />
                          <p className="text-sm text-muted-foreground">
                            {isPreviewLoading
                              ? 'Preparing preview...'
                              : 'Preview will be ready after upload.'}
                          </p>
                          {previewError && (
                            <p className="mt-2 text-xs text-destructive">{previewError}</p>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-muted/20 via-black/60 to-black">
                    <div className="text-center">
                      <FileVideo className="mx-auto mb-2 h-12 w-12 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        {isPreviewLoading
                          ? 'Preparing preview...'
                          : selectedFile?.name || selectedClip?.name || 'Selected source'}
                      </p>
                      {previewError && (
                        <p className="mt-2 text-xs text-destructive">{previewError}</p>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="border-t border-white/10 bg-black/70 p-4 backdrop-blur">
                <div
                  ref={timelineRef}
                  className="relative mb-2 cursor-pointer touch-none"
                  onClick={handleSeek}
                  onPointerDown={handleScrubStart}
                >
                  <div className="h-2 rounded-full bg-muted/40">
                    <div
                      className="h-full rounded-full bg-accent transition-all"
                      style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }}
                    />
                  </div>
                  <div
                    className="absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border border-background bg-accent shadow"
                    style={{ left: `calc(${Math.min(100, Math.max(0, progressPercent))}% - 6px)` }}
                  />
                  {markers.map((marker) => (
                    <div
                      key={marker.id}
                      className={`absolute top-1/2 h-2 w-2 -translate-y-1/2 rounded-full ${marker.highlight}`}
                      style={{ left: `${marker.left}%` }}
                    />
                  ))}
                </div>
                <div className="flex items-center justify-between">
                  <Button
                    size="sm"
                    className="h-8 w-8 rounded-full p-0"
                    onClick={handleTogglePlay}
                    aria-label={isPlaying ? 'Pause video' : 'Play video'}
                    disabled={!showVideoPreview}
                  >
                    {isPlaying ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    <span className="font-mono">
                      {formatTimestamp(currentTime)} / {formatTimestamp(duration)}
                    </span>
                  </span>
                </div>
              </div>

              <div className="border-t border-white/10 bg-black/85 px-4 py-4">
                {selectedFile ? (
                  <>
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
                        <Scissors className="h-3 w-3 text-accent" />
                        Segment selection
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-3 text-xs"
                        onClick={() => {
                          if (!duration) return
                          setSegmentStart(0)
                          setSegmentEnd(duration)
                        }}
                        disabled={!duration}
                      >
                        Full clip
                      </Button>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
                      <span>
                        Start: <span className="font-mono">{formatTimestamp(segmentStart)}</span>
                      </span>
                      <span>
                        End: <span className="font-mono">{formatTimestamp(segmentEnd)}</span>
                      </span>
                      <span>
                        Length:{' '}
                        <span className="font-mono">{formatTimestamp(segmentDuration)}</span>
                      </span>
                    </div>

                    <div ref={segmentTrackRef} className="relative mt-3 h-4">
                      <div className="absolute inset-0 rounded-full bg-muted/40" />
                      <div
                        className="absolute inset-y-0 rounded-full bg-accent/40"
                        style={{
                          left: `${Math.min(100, Math.max(0, segmentStartPct))}%`,
                          right: `${Math.min(100, Math.max(0, 100 - segmentEndPct))}%`,
                        }}
                      />
                      <button
                        type="button"
                        className={`segment-handle cursor-ew-resize focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-black ${duration ? '' : 'segment-handle--disabled'
                          }`}
                        style={{ left: `calc(${segmentStartPct}% - 7px)` }}
                        onPointerDown={handleSegmentDragStart('start')}
                        aria-label="Drag segment start"
                        disabled={!duration}
                      />
                      <button
                        type="button"
                        className={`segment-handle cursor-ew-resize focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-black ${duration ? '' : 'segment-handle--disabled'
                          }`}
                        style={{ left: `calc(${segmentEndPct}% - 7px)` }}
                        onPointerDown={handleSegmentDragStart('end')}
                        aria-label="Drag segment end"
                        disabled={!duration}
                      />
                    </div>

                    <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
                      <span>
                        {isFullSegment
                          ? 'Full clip selected. Uploading original.'
                          : 'Only the highlighted segment uploads.'}
                      </span>
                      {isTrimming && <span className="text-accent">Trimming locally...</span>}
                    </div>

                    {trimWarning && (
                      <div className="mt-2 text-[11px] text-amber-400">{trimWarning}</div>
                    )}
                    {isProbingDuration && (
                      <div className="mt-2 text-[11px] text-muted-foreground">
                        Reading duration for segment selection...
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                    <span>Recorded mode analyzes the selected CCTV clip directly.</span>
                    <span className="text-accent">Clip: {selectedClip?.name || '--'}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </div>

        <aside className="flex flex-col gap-4">
          {hasResults && report?.tool_used && (
            <div className={`${panelClass}`}>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Analysis Result
              </p>
              <p className="mt-3 text-sm leading-relaxed text-foreground">
                Tool: <span className="font-semibold">{report.tool_used}</span>
              </p>
              <p className="mt-1 text-sm text-foreground">
                Events found: <span className="font-semibold">{report.events_found || results.length}</span>
              </p>
              {report.intent_config && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Target: {report.intent_config.target}
                </p>
              )}
              {report.verification && (
                <div className="mt-3 rounded-lg border border-white/10 bg-white/5 p-2 text-xs">
                  <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
                    Verification
                  </p>
                  <div className="flex gap-3 text-foreground">
                    <span>Total: {report.verification.total}</span>
                    <span>Kept: <span className="text-emerald-400">{report.verification.kept}</span></span>
                    <span>Dropped: <span className="text-rose-400">{report.verification.dropped}</span></span>
                  </div>
                </div>
              )}
              {plateList.length > 0 && (
                <div className="mt-4">
                  <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Plates ({plateList.length})
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {plateList.map((plate, index) => (
                      <span
                        key={`${plate}-${index}`}
                        className="rounded-full border border-white/10 bg-white/5 px-2 py-1 font-mono text-[11px] text-foreground/90"
                      >
                        {plate}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {!hasResults && isTextMode ? (
            <div className={`${panelClass}`}>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Summary Response
              </p>
              <p className="mt-3 text-sm leading-relaxed text-foreground">
                {textAnswer || 'No summary was generated for this query.'}
              </p>
            </div>
          ) : selectedEvent ? (
            <div className={`${panelClass}`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    Selected Event
                  </p>
                  <p className="text-sm font-semibold text-foreground">
                    {selectedEvent.description}
                  </p>
                </div>
                <Button variant="outline" size="sm" onClick={() => handleSelectEvent(selectedEvent)}>
                  Jump to {selectedEvent.timestamp}
                </Button>
              </div>
              <div className="mt-4 space-y-3">
                <div className={`${panelInsetClass} p-3 text-xs text-muted-foreground`}>
                  <p className="text-[10px] uppercase tracking-wide">Details</p>
                  <div className="mt-2 grid gap-1">
                    <p>
                      Type: <span className="text-foreground">{selectedEvent.type}</span>
                    </p>
                    {selectedEvent.plateText && (
                      <p>
                        Plate:{' '}
                        <span className="font-mono text-foreground">
                          {selectedEvent.plateText}
                        </span>
                      </p>
                    )}
                    {selectedEvent.verification !== null && (
                      <p>
                        Verification:{' '}
                        <span className="text-foreground">
                          {selectedEvent.verified ? 'Confirmed' : 'Unverified'}
                        </span>
                      </p>
                    )}
                    {selectedEvent.confidence && (
                      <p>
                        Confidence:{' '}
                        <span className="text-foreground">{selectedEvent.confidence}%</span>
                      </p>
                    )}
                    <p>
                      Time:{' '}
                      <span className="font-mono text-foreground">
                        {selectedEvent.timestamp}
                      </span>
                    </p>
                  </div>
                </div>
                <div className={`${panelInsetClass} p-3`}>
                  <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Evidence
                  </p>
                  {selectedEvent.thumbnail ? (
                    <BBoxImage
                      src={selectedEvent.thumbnail}
                      bbox={selectedEvent.bbox}
                      alt="Evidence"
                      className="mt-3 aspect-video w-full rounded-md border border-white/10 bg-black"
                      imgClassName="h-full w-full object-contain"
                    />
                  ) : thumbnailLoadingId === selectedEvent.id ? (
                    <div className="mt-3 flex aspect-video items-center justify-center rounded-md border border-white/10 text-xs text-muted-foreground">
                      Loading thumbnail...
                    </div>
                  ) : (
                    <div className="mt-3 flex aspect-video items-center justify-center rounded-md border border-white/10 text-xs text-muted-foreground">
                      No thumbnail provided. Timestamp: {selectedEvent.timestamp}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className={`${panelSoftClass} border-dashed text-sm text-muted-foreground`}>
              No event results available for this query.
            </div>
          )}

          {results.length > 0 && (
            <div className={`${panelClass}`}>
              <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
                <h3 className="text-sm font-semibold text-foreground">
                  EVENTS ({results.length})
                </h3>
                <span className="text-xs text-muted-foreground">Click to review</span>
              </div>
              <div className="max-h-[360px] overflow-y-auto divide-y divide-white/10">
                {results.map((result) => (
                  <button
                    key={result.id}
                    type="button"
                    onClick={() => handleOpenEventDialog(result)}
                    className={`group w-full cursor-pointer px-5 py-4 text-left transition-colors hover:bg-white/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-black ${selectedEvent?.id === result.id ? 'bg-white/5' : ''
                      }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="relative flex h-12 w-16 flex-shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/10 bg-black/50">
                        {result.thumbnail ? (
                          <BBoxImage
                            src={result.thumbnail}
                            bbox={result.bbox}
                            alt="Event"
                            className="h-full w-full bg-black/60"
                            imgClassName="h-full w-full object-contain"
                          />
                        ) : thumbnailLoadingId === result.id ? (
                          <span className="text-[10px] text-muted-foreground">Loading...</span>
                        ) : (
                          <FileVideo className="h-5 w-5 text-muted-foreground" />
                        )}
                        <div className="absolute right-1 top-1 rounded bg-black/70 px-1 font-mono text-[9px] text-white/70">
                          {result.timestamp}
                        </div>
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="line-clamp-1 text-sm font-medium text-foreground">
                            {result.plateText || result.description}
                          </p>
                          {result.vehicleType && (
                            <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-[10px] text-sky-300 uppercase">
                              {result.vehicleType}
                            </span>
                          )}
                          {result.plateConfidence != null && (
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[10px] ${result.plateConfidence >= 0.8
                                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                                : result.plateConfidence >= 0.5
                                  ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                                  : 'border-red-500/30 bg-red-500/10 text-red-300'
                                }`}
                              title={result.plateOriginal ? `OCR corrected: ${result.plateOriginal} → ${result.plateText}` : undefined}
                            >
                              {result.plateConfidence >= 0.8 ? '✓ High' : result.plateConfidence >= 0.5 ? '⚠ Medium' : '✗ Low'}
                            </span>
                          )}
                          {result.plateOriginal && (
                            <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-2 py-0.5 text-[10px] text-violet-300" title={`Raw OCR: ${result.plateOriginal}`}>
                              corrected
                            </span>
                          )}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                          <span className="inline-flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            <span className="font-mono">{result.timestamp}</span>
                          </span>
                          {result.trackId != null && (
                            <span className="font-mono text-[10px] text-muted-foreground/70">
                              Track #{result.trackId}
                            </span>
                          )}
                          <span
                            className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide ${getTypePillClass(
                              result.type
                            )}`}
                          >
                            {result.type || 'event'}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span className="hidden sm:inline">Open</span>
                        <div className="flex h-8 w-8 items-center justify-center rounded-full border border-white/10 bg-white/5 transition group-hover:border-white/30 group-hover:text-foreground">
                          {getTypeIcon(result.type)}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>

      {isEventDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6">
          <button
            type="button"
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsEventDialogOpen(false)}
            aria-label="Close event details"
          />
          <div className="relative z-10 w-[min(94vw,960px)] max-h-[85vh] overflow-hidden rounded-2xl border border-white/10 bg-black/95 p-6 text-foreground shadow-[0_0_50px_rgba(0,0,0,0.6)]">
            <button
              type="button"
              className="absolute right-4 top-4 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] text-muted-foreground transition hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
              onClick={() => setIsEventDialogOpen(false)}
            >
              Close
            </button>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-1">
                <p className="text-lg font-semibold text-foreground">
                  {dialogEvent?.description || 'Event details'}
                </p>
                <p className="text-sm text-muted-foreground">
                  Review evidence and jump to the exact timestamp.
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-foreground/80">
                    <Clock className="h-3 w-3" />
                    <span className="font-mono">{dialogEvent?.timestamp || '--:--:--'}</span>
                  </span>
                  <span
                    className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide ${getTypePillClass(
                      dialogEvent?.type
                    )}`}
                  >
                    {dialogEvent?.type || 'event'}
                  </span>
                  {dialogEvent?.confidence && (
                    <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-foreground/80">
                      {dialogEvent.confidence}% confidence
                    </span>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => dialogEvent && handleSelectEvent(dialogEvent)}
                >
                  Jump to timestamp
                </Button>
              </div>
            </div>
            <div className="mt-5 grid gap-4 sm:grid-cols-[1.2fr_0.8fr]">
              <div className={`${panelInsetClass} p-3`}>
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Evidence
                </p>
                {dialogEvent?.thumbnail ? (
                  <BBoxImage
                    src={dialogEvent.thumbnail}
                    bbox={dialogEvent.bbox}
                    alt="Evidence"
                    className="mt-3 aspect-video w-full rounded-md border border-white/10 bg-black"
                    imgClassName="h-full w-full object-contain"
                  />
                ) : (
                  <div className="mt-3 flex aspect-video items-center justify-center rounded-md border border-white/10 text-xs text-muted-foreground">
                    No snapshot available
                  </div>
                )}
              </div>
              <div className={`${panelInsetClass} p-3 text-xs text-muted-foreground`}>
                <p className="text-[10px] uppercase tracking-wide">Details</p>
                <div className="mt-2 grid gap-2">
                  <div>
                    Type:{' '}
                    <span className="text-foreground">{dialogEvent?.type || 'n/a'}</span>
                  </div>
                  {dialogEvent?.plateText && (
                    <div>
                      Plate:{' '}
                      <span className="font-mono text-foreground">
                        {dialogEvent.plateText}
                      </span>
                      {dialogEvent.plateFormatValid && (
                        <span className="ml-1 text-emerald-400">✓</span>
                      )}
                    </div>
                  )}
                  {dialogEvent?.plateOriginal && (
                    <div>
                      Raw OCR:{' '}
                      <span className="font-mono text-foreground/60 line-through">
                        {dialogEvent.plateOriginal}
                      </span>
                      <span className="ml-1 text-violet-400 text-[10px]">auto-corrected</span>
                    </div>
                  )}
                  {dialogEvent?.vehicleType && (
                    <div>
                      Vehicle:{' '}
                      <span className="text-foreground capitalize">{dialogEvent.vehicleType}</span>
                    </div>
                  )}
                  {dialogEvent?.plateConfidence != null && (
                    <div>
                      Plate confidence:{' '}
                      <span className={
                        dialogEvent.plateConfidence >= 0.8 ? 'text-emerald-400' :
                          dialogEvent.plateConfidence >= 0.5 ? 'text-amber-400' :
                            'text-red-400'
                      }>
                        {Math.round(dialogEvent.plateConfidence * 100)}%
                      </span>
                      {dialogEvent.plateFormat && (
                        <span className="ml-1 text-foreground/50">({dialogEvent.plateFormat})</span>
                      )}
                    </div>
                  )}
                  {dialogEvent?.confidence && !dialogEvent?.plateConfidence && (
                    <div>
                      Confidence:{' '}
                      <span className="text-foreground">{dialogEvent.confidence}%</span>
                    </div>
                  )}
                  {showSegmentTimestamp && (
                    <div>
                      Segment time:{' '}
                      <span className="font-mono text-foreground">
                        {dialogEvent.segmentTimestamp}
                      </span>
                    </div>
                  )}
                  {dialogEvent && dialogEvent.verification !== null && (
                    <div>
                      Verification:{' '}
                      <span className="text-foreground">
                        {dialogEvent.verified ? 'Confirmed' : 'Unverified'}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChange}
      />
      <input
        ref={imageInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleImageChange}
      />
    </div>
  )
}
