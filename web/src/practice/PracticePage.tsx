import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router'
import { getChart } from '../api/analyze'
import { useApiClient } from '../api/context'
import { ApiError } from '../api/http'
import type { Chart } from '../api/types'
import { applyOverrides } from '../grid/effective'
import { buildGrid } from '../grid/grid'
import { useI18n } from '../i18n/context'
import type { Rate, TransportError } from '../transport/types'
import { useTransport } from '../transport/useTransport'
import { ensureAudioContext } from './audio'
import { PracticeController } from './controller'
import { ChordModeToggle } from './ChordModeToggle'
import { entitlementsFor, type Feature } from './entitlements'
import { LoopControls } from './LoopControls'
import { MetronomePanel } from './MetronomePanel'
import { SmartGrid } from './SmartGrid'
import { TransportBar } from './TransportBar'
import { UnlockModal } from './UnlockModal'
import { UpgradePrompt } from './UpgradePrompt'
import { useAuth } from '../auth/context'
import { useChordMode } from './useChordMode'
import { useMetronome } from './useMetronome'
import { usePlayhead } from './usePlayhead'
import { useShortcuts } from './useShortcuts'

/** The practice workspace for one chart: the grid, the player and what it needs to stay in step. */
export default function PracticePage() {
  const { id = '' } = useParams()
  const client = useApiClient()
  const { t } = useI18n()
  const { data: chart, error, isPending } = useQuery({
    queryKey: ['chart', id],
    queryFn: () => getChart(client, id),
    retry: false,
  })

  if (isPending) {
    return (
      <p role="status" className="text-slate-400">
        {t.chart.loading}
      </p>
    )
  }
  // A failed background refresh keeps the chart that is already on screen: only a missing one is an error.
  if (!chart && error instanceof ApiError && error.status === 404) {
    return (
      <div className="text-slate-300">
        <p>{t.chart.notFound}</p>
        <Link to="/library" className="mt-2 inline-block text-brand-purple-light underline">
          {t.chart.back}
        </Link>
      </div>
    )
  }
  if (!chart) {
    return (
      <p role="alert" className="rounded-lg bg-red-500/10 px-3 py-2 text-red-300">
        {t.chart.unavailable}
      </p>
    )
  }
  // A different chart starts from a clean slate (grid, transport, unlock prompt).
  return <Practice key={chart.id} chart={chart} />
}

function Practice({ chart }: { chart: Chart }) {
  const { t } = useI18n()
  const [mount, setMount] = useState<HTMLElement | null>(null)
  // The autoplay error the person has already answered; a new refusal is a new object.
  const [answered, setAnswered] = useState<TransportError | null>(null)
  const gridRef = useRef<HTMLDivElement>(null)

  const grid = useMemo(
    () => buildGrid(chart.metadata, applyOverrides(chart.ai_events, chart.overrides)),
    [chart],
  )
  const [attempt, setAttempt] = useState(0)
  const resumeAt = useRef<number | null>(null)
  const { transport, state, error } = useTransport(
    { id: chart.id, playback: chart.source.playback },
    mount,
    undefined,
    attempt,
  )
  useEffect(() => {
    if (transport && state === 'ready' && resumeAt.current !== null) {
      transport.seek(resumeAt.current) // a retry picks the song up where it stopped
      resumeAt.current = null
    }
  }, [transport, state])
  const retry = () => {
    resumeAt.current = transport?.currentTimeMs() ?? 0
    setAttempt((n) => n + 1)
  }
  const controller = useMemo(() => (transport ? new PracticeController(transport, grid) : null), [transport, grid])
  usePlayhead(transport, grid, gridRef, { onFrame: (ms) => controller?.frame(ms) })
  const [rate, setRate] = useState<Rate>(1)
  const [mode, setMode] = useChordMode()
  const { user } = useAuth()
  const entitlements = entitlementsFor(user?.role ?? 'guest')
  const [locked, setLocked] = useState<Feature | null>(null)
  const metronome = useMetronome({
    transport,
    grid,
    controller,
    rate,
    tempoBpm: chart.metadata.beat_grid.tempo_bpm,
    beatsPerBar: chart.metadata.beat_grid.beats_per_bar,
  })
  const [loopBars, setLoopBars] = useState<{ from: number; to: number } | null>(null)

  const start = useCallback(async () => {
    if (!transport) return
    const ctx = ensureAudioContext()
    await ctx?.resume().catch(() => {})
    // A refusal or failure is reported through the transport's error event.
    if (transport.state !== 'playing' && !(await metronome.runCountIn())) return
    await transport.play().catch(() => {})
  }, [transport, metronome])

  const toggle = () => {
    if (state === 'playing') transport?.pause()
    else void start()
  }

  useShortcuts(controller, toggle)

  const selectLoop = (a: number, b: number) => {
    if (!controller?.loop.setFromBars(grid, a, b)) return
    const last = grid.bars.length - 1
    setLoopBars({ from: Math.max(Math.min(a, b), 0), to: Math.min(Math.max(a, b), last) })
  }
  const clearLoop = () => {
    controller?.loop.clear()
    setLoopBars(null)
  }
  const changeRate = (next: Rate) => {
    if (next !== 1 && !entitlements.speed) {
      setLocked('speed')
      return
    }
    setLocked(null)
    transport?.setRate(next)
    setRate(next)
  }

  const { key, beat_grid: beats } = chart.metadata
  const kind = chart.source.playback.kind
  const embedded = kind === 'youtube' || kind === 'soundcloud'
  const canPlay = state === 'ready' || state === 'paused' || state === 'ended' || state === 'playing'

  return (
    <div className="mx-auto max-w-4xl">
      <h2 className="font-display text-2xl font-bold text-white">{chart.title}</h2>
      <p className="mt-2 text-slate-400">
        {t.chart.key}: {key.tonic} {key.mode} · {t.chart.tempo}: {Math.round(beats.tempo_bpm)} {t.chart.bpm}
      </p>

      <div
        ref={setMount}
        data-testid="media-frame"
        hidden={!embedded}
        className="mt-4 aspect-video w-full max-w-xl overflow-hidden rounded-xl bg-black [&>*]:h-full [&>*]:w-full"
      />

      {error && error.code !== 'autoplay_blocked' && (error.code === 'embed_blocked' || embedded) ? (
        <div role="alert" className="mt-4 rounded-lg bg-amber-500/10 px-3 py-2 text-amber-200">
          <p>{t.practice.embedBlocked}</p>
          <Link to="/workspace" className="mt-1 inline-block underline">
            {t.practice.uploadInstead}
          </Link>
        </div>
      ) : error && error.code !== 'autoplay_blocked' ? (
        <div role="alert" className="mt-4 rounded-lg bg-red-500/10 px-3 py-2 text-red-300">
          <p>{t.practice.playbackFailed}</p>
          <button type="button" onClick={retry} className="mt-1 underline">
            {t.practice.retry}
          </button>
        </div>
      ) : null}

      <div className="mt-4 space-y-2">
        <TransportBar
          transport={transport}
          playing={state === 'playing'}
          canPlay={canPlay}
          durationMs={transport?.durationMs() ?? chart.metadata.duration_ms}
          rate={rate}
          onToggle={toggle}
          onRate={changeRate}
          speedLocked={!entitlements.speed}
        />
        <LoopControls range={loopBars} onClear={clearLoop} />
        <ChordModeToggle mode={mode} onChange={setMode} />
      </div>

      <div className="mt-6">
        <SmartGrid
          grid={grid}
          mode={mode}
          rootRef={gridRef}
          loop={loopBars}
          onSeek={(ms) => controller?.seek(ms)}
          onSelect={selectLoop}
        />
      </div>

      <div className="mt-6">
        <MetronomePanel metronome={metronome} entitlements={entitlements} onLocked={setLocked} />
        {locked && <UpgradePrompt feature={locked} onDismiss={() => setLocked(null)} />}
      </div>

      {error?.code === 'autoplay_blocked' && error !== answered && (
        <UnlockModal
          onUnlock={() => {
            setAnswered(error)
            void start()
          }}
        />
      )}
    </div>
  )
}
