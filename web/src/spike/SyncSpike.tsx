/**
 * Dev-only harness for the NFR-01 measurement (route /__spike/sync, never in a production build).
 *
 * It plays a click track generated in memory through the real HtmlAudioTransport, lights bars with
 * the real SmartGrid + usePlayhead, and on every frame compares the lit bar with the bar the audio
 * clock says is playing. The numbers land in `window.__syncStats` for Playwright.
 *
 * What it measures: the gap between the audio element's clock and the bar lit in the DOM, i.e. the
 * synchronisation logic. What it cannot see: the delay between the DOM write and photons on the
 * screen, and the output latency of the audio device.
 */
import { useMemo, useRef, useState } from 'react'
import type { AnalysisMetadata, ChordEvent } from '../api/types'
import { buildGrid } from '../grid/grid'
import { SmartGrid } from '../practice/SmartGrid'
import { usePlayhead } from '../practice/usePlayhead'
import { HtmlAudioTransport } from '../transport/html'
import { flipLags, offsetMs, summarize, type Sample, type Stats } from './stats'

declare global {
  interface Window {
    __syncStats?: Stats
  }
}

const RATE = 8000
const OFFSET_MS = 250
const CHORDS = [
  ['Cmaj7', 'C', 'major', 'maj7'],
  ['Am7', 'A', 'minor', 'min7'],
  ['Dm7', 'D', 'minor', 'min7'],
  ['G7', 'G', 'dominant', 'dom7'],
] as const

function metadata(seconds: number): AnalysisMetadata {
  return {
    duration_ms: seconds * 1000,
    key: { tonic: 'C', mode: 'major', confidence: 1 },
    beat_grid: {
      tempo_bpm: 120,
      tempo_confidence: 1,
      time_signature: { numerator: 4, denominator: 4 },
      beats_per_bar: 4,
      first_beat_offset_ms: OFFSET_MS,
      downbeats_ms: null,
    },
    model_version: 'spike',
    pipeline_version: 'spike',
  }
}

/** One chord per bar, like a real chart of this length. */
function chordsFor(seconds: number): ChordEvent[] {
  const barMs = 2000
  const count = Math.floor((seconds * 1000 - OFFSET_MS) / barMs)
  return Array.from({ length: count }, (_, i) => {
    const [symbol, root, quality, extension] = CHORDS[i % CHORDS.length]
    return {
      event_id: `e${i}`,
      start_ms: OFFSET_MS + i * barMs,
      end_ms: OFFSET_MS + (i + 1) * barMs,
      chord: { root, quality, extension, bass: null },
      symbol,
      confidence: null,
      is_user_edited: false,
    }
  })
}

/** A WAV of silence with a short tick on every beat, the first of each bar higher. */
function clickTrack(seconds: number): Blob {
  const samples = new Int16Array(RATE * seconds)
  for (let beat = 0; OFFSET_MS + beat * 500 < seconds * 1000; beat++) {
    const start = Math.round(((OFFSET_MS + beat * 500) * RATE) / 1000)
    const hz = beat % 4 === 0 ? 2000 : 1000
    for (let i = 0; i < 160 && start + i < samples.length; i++) {
      samples[start + i] = Math.round(9000 * Math.sin((2 * Math.PI * hz * i) / RATE) * (1 - i / 160))
    }
  }
  const header = new DataView(new ArrayBuffer(44))
  const text = (at: number, value: string) => [...value].forEach((c, i) => header.setUint8(at + i, c.charCodeAt(0)))
  text(0, 'RIFF')
  header.setUint32(4, 36 + samples.byteLength, true)
  text(8, 'WAVEfmt ')
  header.setUint32(16, 16, true)
  header.setUint16(20, 1, true)
  header.setUint16(22, 1, true)
  header.setUint32(24, RATE, true)
  header.setUint32(28, RATE * 2, true)
  header.setUint16(32, 2, true)
  header.setUint16(34, 16, true)
  text(36, 'data')
  header.setUint32(40, samples.byteLength, true)
  return new Blob([header, samples], { type: 'audio/wav' })
}

export default function SyncSpike() {
  const seconds = Math.min(Math.max(Number(new URLSearchParams(location.search).get('seconds')) || 30, 5), 600)
  const grid = useMemo(() => buildGrid(metadata(seconds), chordsFor(seconds)), [seconds])
  const root = useRef<HTMLDivElement>(null)
  const [transport, setTransport] = useState<HtmlAudioTransport | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)

  usePlayhead(transport, grid, root)

  async function start() {
    const url = URL.createObjectURL(clickTrack(seconds))
    const player = new HtmlAudioTransport(new Audio(), async () => url)
    await player.load()
    setTransport(player)
    await player.play()

    const samples: Sample[] = []
    const sample = (at: number) => {
      const audioMs = player.currentTimeMs()
      const lit = root.current?.querySelector('[data-current="true"]')
      const shown = lit ? Number(lit.getAttribute('data-bar')) : -1
      samples.push({ at, audioMs, shown, offset: offsetMs(grid, audioMs, shown) })
      if (audioMs >= seconds * 1000 - 300 || player.state === 'ended') {
        const result = summarize(samples, flipLags(grid, samples))
        window.__syncStats = result
        setStats(result)
        player.destroy()
        return
      }
      requestAnimationFrame(sample)
    }
    requestAnimationFrame(sample) // registered after the playhead's loop, so it sees the result of its work
  }

  return (
    <main className="p-6">
      <h1 className="mb-4 text-xl font-bold text-white">NFR-01 sync spike ({seconds} s)</h1>
      <button type="button" onClick={() => void start()} className="mb-4 rounded bg-brand-purple px-4 py-2 text-white">
        Start
      </button>
      <pre data-testid="stats" className="mb-4 text-slate-300">
        {stats ? JSON.stringify(stats) : ''}
      </pre>
      <SmartGrid grid={grid} mode="precise" onSeek={() => {}} rootRef={root} />
    </main>
  )
}
