import { useCallback, useEffect, useRef, useState } from 'react'
import type { Grid } from '../grid/grid'
import { clampBpm, manualBeats, songBeats } from '../metronome/beats'
import { scheduleClick, type Timbre } from '../metronome/click'
import { MetronomeScheduler, type PlaybackClock } from '../metronome/scheduler'
import { TapTempo } from '../metronome/tap'
import type { Rate, Transport } from '../transport/types'
import { ensureAudioContext } from './audio'
import type { PracticeController } from './controller'

export type MetronomeMode = 'manual' | 'song'

export interface MetronomeOptions {
  transport: Transport | null
  grid: Grid
  controller: PracticeController | null
  rate: Rate
  /** Tempo and bar length of the analysed song: where Manual mode starts from. */
  tempoBpm: number
  beatsPerBar: number
}

const COUNT_IN_LEAD_S = 0.05 // so the first click is not scheduled in the past

export function useMetronome({ transport, grid, controller, rate, tempoBpm, beatsPerBar: songBeatsPerBar }: MetronomeOptions) {
  const [enabled, setEnabledState] = useState(false)
  const [mode, setMode] = useState<MetronomeMode>('manual')
  const [bpm, setBpmState] = useState(() => Math.round(clampBpm(tempoBpm)))
  const [beatsPerBar, setBeatsPerBarState] = useState(songBeatsPerBar)
  const [timbre, setTimbre] = useState<Timbre>('woodblock')
  const [volume, setVolume] = useState(0.8)
  const [countInOn, setCountInOn] = useState(false)

  const schedulerRef = useRef<MetronomeScheduler | null>(null)
  const rateRef = useRef(rate)
  const tap = useRef(new TapTempo())
  const sound = useRef({ timbre, volume })
  const countIn = useRef<{ timer: ReturnType<typeof setTimeout>; done: (go: boolean) => void } | null>(null)

  useEffect(() => {
    rateRef.current = rate
    schedulerRef.current?.reset() // a new speed moves every upcoming beat
  }, [rate])

  useEffect(() => {
    sound.current = { timbre, volume }
    schedulerRef.current?.setTimbre(timbre)
    schedulerRef.current?.setVolume(volume)
  }, [timbre, volume])

  // Anything that moves the playhead other than playing makes the queued clicks wrong.
  useEffect(() => controller?.onJump(() => schedulerRef.current?.reset()), [controller])

  useEffect(() => {
    if (!enabled) return
    const ctx = ensureAudioContext()
    if (!ctx) return
    let clock: PlaybackClock
    if (mode === 'song' && transport) {
      clock = {
        sample: () => ({
          songMs: transport.currentTimeMs(),
          ctxTime: ctx.currentTime,
          rate: rateRef.current,
          playing: transport.state === 'playing',
        }),
      }
    } else {
      const origin = ctx.currentTime
      clock = {
        sample: () => ({ songMs: (ctx.currentTime - origin) * 1000, ctxTime: ctx.currentTime, rate: 1, playing: true }),
      }
    }
    const source = mode === 'song' ? songBeats(grid) : manualBeats(bpm, beatsPerBar)
    const scheduler = new MetronomeScheduler(ctx, source, clock, sound.current)
    schedulerRef.current = scheduler
    scheduler.start()
    return () => {
      scheduler.stop()
      if (schedulerRef.current === scheduler) schedulerRef.current = null
    }
  }, [enabled, mode, bpm, beatsPerBar, grid, transport])

  // Leaving the page cancels a count-in that is still running.
  useEffect(
    () => () => {
      const pending = countIn.current
      if (pending) {
        clearTimeout(pending.timer)
        pending.done(false)
      }
    },
    [],
  )

  /** Needs a user gesture: the audio context is created and resumed here. */
  const setEnabled = useCallback(async (on: boolean) => {
    if (on) {
      const ctx = ensureAudioContext()
      if (!ctx) return
      await ctx.resume().catch(() => {})
    }
    setEnabledState(on)
  }, [])

  const setBpm = useCallback((value: number) => setBpmState(Math.round(clampBpm(value) * 10) / 10), [])
  const setBeatsPerBar = useCallback((value: number) => {
    if (Number.isFinite(value)) setBeatsPerBarState(Math.min(Math.max(Math.round(value), 1), 12))
  }, [])

  const tapTempo = useCallback(() => {
    const estimate = tap.current.tap(performance.now())
    if (estimate !== null) setBpm(estimate)
  }, [setBpm])

  /** One bar of clicks before the song starts; resolves true when it is time to play, false if cancelled. */
  const runCountIn = useCallback(async (): Promise<boolean> => {
    if (!countInOn) return true
    const ctx = ensureAudioContext()
    if (!ctx) return true
    await ctx.resume().catch(() => {})
    const tempo = mode === 'song' ? tempoBpm : bpm
    const perBar = mode === 'song' ? songBeatsPerBar : beatsPerBar
    const beatS = 60 / tempo / rateRef.current
    const first = ctx.currentTime + COUNT_IN_LEAD_S
    for (let i = 0; i < perBar; i++) {
      scheduleClick(ctx, first + i * beatS, { accent: i === 0, timbre: sound.current.timbre, volume: sound.current.volume })
    }
    return new Promise<boolean>((resolve) => {
      const done = (go: boolean) => {
        countIn.current = null
        resolve(go)
      }
      const timer = setTimeout(() => done(true), (COUNT_IN_LEAD_S + perBar * beatS) * 1000)
      countIn.current = { timer, done }
    })
  }, [countInOn, mode, bpm, beatsPerBar, tempoBpm, songBeatsPerBar])

  return {
    enabled,
    setEnabled,
    mode,
    setMode,
    bpm,
    setBpm,
    beatsPerBar,
    setBeatsPerBar,
    timbre,
    setTimbre,
    volume,
    setVolume,
    tapTempo,
    countInOn,
    setCountInOn,
    runCountIn,
  }
}

export type MetronomeState = ReturnType<typeof useMetronome>
