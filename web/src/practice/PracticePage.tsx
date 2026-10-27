import { useQuery } from '@tanstack/react-query'
import { useCallback, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router'
import { getChart } from '../api/analyze'
import { useApiClient } from '../api/context'
import { ApiError } from '../api/http'
import type { Chart } from '../api/types'
import type { ChordMode } from '../format/chord'
import { applyOverrides } from '../grid/effective'
import { buildGrid } from '../grid/grid'
import { useI18n } from '../i18n/context'
import type { TransportError } from '../transport/types'
import { useTransport } from '../transport/useTransport'
import { ensureAudioContext } from './audio'
import { SmartGrid } from './SmartGrid'
import { UnlockModal } from './UnlockModal'
import { usePlayhead } from './usePlayhead'

const MODE: ChordMode = 'precise'

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
  if (error instanceof ApiError && error.status === 404) {
    return (
      <div className="text-slate-300">
        <p>{t.chart.notFound}</p>
        <Link to="/library" className="mt-2 inline-block text-brand-purple-light underline">
          {t.chart.back}
        </Link>
      </div>
    )
  }
  if (error || !chart) {
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
  const { transport, state, error } = useTransport({ id: chart.id, playback: chart.source.playback }, mount)
  usePlayhead(transport, grid, gridRef)

  const start = useCallback(async () => {
    if (!transport) return
    const ctx = ensureAudioContext()
    await ctx?.resume().catch(() => {})
    // A refusal or failure is reported through the transport's error event.
    await transport.play().catch(() => {})
  }, [transport])

  const toggle = () => {
    if (state === 'playing') transport?.pause()
    else void start()
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

      {error?.code === 'embed_blocked' ? (
        <div role="alert" className="mt-4 rounded-lg bg-amber-500/10 px-3 py-2 text-amber-200">
          <p>{t.practice.embedBlocked}</p>
          <Link to="/workspace" className="mt-1 inline-block underline">
            {t.practice.uploadInstead}
          </Link>
        </div>
      ) : error && error.code !== 'autoplay_blocked' ? (
        <p role="alert" className="mt-4 rounded-lg bg-red-500/10 px-3 py-2 text-red-300">
          {t.practice.playbackFailed}
        </p>
      ) : null}

      <div className="mt-4">
        <button
          type="button"
          disabled={!canPlay}
          onClick={toggle}
          className="rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white hover:opacity-90 disabled:opacity-40"
        >
          {state === 'playing' ? t.practice.pause : t.practice.play}
        </button>
      </div>

      <div className="mt-6">
        <SmartGrid grid={grid} mode={MODE} rootRef={gridRef} onSeek={(ms) => transport?.seek(ms)} />
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
