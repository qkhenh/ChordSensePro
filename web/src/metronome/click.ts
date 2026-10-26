export type Timbre = 'woodblock' | 'beep'

export interface ClickOptions {
  accent: boolean
  timbre: Timbre
  /** 0 to 1. */
  volume: number
}

interface Voice {
  type: OscillatorType
  normal: number
  accent: number
  seconds: number
}

const VOICES: Record<Timbre, Voice> = {
  woodblock: { type: 'triangle', normal: 1000, accent: 1600, seconds: 0.045 },
  beep: { type: 'sine', normal: 880, accent: 1760, seconds: 0.06 },
}
const ATTACK_SECONDS = 0.002
const ACCENT_LEVEL = 1
const NORMAL_LEVEL = 0.7

/**
 * Queue one click on the audio clock at `when` (the context's own time, in seconds). Returns a
 * function that cancels it if it has not sounded yet. Used both live and with an
 * OfflineAudioContext, which is how the 10 ms accuracy of NFR-04 is measured.
 */
export function scheduleClick(ctx: BaseAudioContext, when: number, opts: ClickOptions): () => void {
  const peak = Math.max(0, Math.min(1, opts.volume)) * (opts.accent ? ACCENT_LEVEL : NORMAL_LEVEL)
  if (peak === 0) return () => {}

  const voice = VOICES[opts.timbre]
  const osc = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.type = voice.type
  osc.frequency.value = opts.accent ? voice.accent : voice.normal
  gain.gain.setValueAtTime(0, when)
  gain.gain.linearRampToValueAtTime(peak, when + ATTACK_SECONDS)
  gain.gain.exponentialRampToValueAtTime(0.0001, when + voice.seconds)
  osc.connect(gain)
  gain.connect(ctx.destination)
  osc.start(when)
  osc.stop(when + voice.seconds + 0.02)
  return () => {
    try {
      osc.stop(0)
    } catch {
      /* already finished */
    }
  }
}
