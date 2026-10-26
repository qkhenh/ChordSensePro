import type { Grid } from '../grid/grid'

export const MIN_BPM = 30
export const MAX_BPM = 280

export interface Beat {
  /** Position in the song, in milliseconds. */
  ms: number
  /** The first beat of a bar. */
  accent: boolean
}

export interface BeatSource {
  /** The beats in the half-open window [fromMs, toMs). */
  beatsBetween(fromMs: number, toMs: number): Beat[]
}

export function clampBpm(bpm: number): number {
  return Math.min(Math.max(bpm, MIN_BPM), MAX_BPM)
}

/** The beats of the analysed song, following its bars exactly (Song Sync mode). */
export function songBeats(grid: Grid): BeatSource {
  const all: Beat[] = grid.bars.flatMap((bar) =>
    bar.beatsMs.map((ms, i) => ({ ms, accent: i === 0 })),
  )
  return {
    beatsBetween(fromMs, toMs) {
      let lo = 0
      let hi = all.length
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (all[mid].ms < fromMs) lo = mid + 1
        else hi = mid
      }
      const out: Beat[] = []
      for (let i = lo; i < all.length && all[i].ms < toMs; i++) out.push(all[i])
      return out
    },
  }
}

/** A free-running metronome with no song behind it (Manual mode). */
export function manualBeats(bpm: number, beatsPerBar: number): BeatSource {
  const interval = 60_000 / clampBpm(bpm)
  return {
    beatsBetween(fromMs, toMs) {
      const out: Beat[] = []
      // The tiny allowance keeps a beat that floating point puts a hair before `fromMs`.
      for (let k = Math.max(Math.ceil(fromMs / interval - 1e-9), 0); k * interval < toMs; k++) {
        out.push({ ms: k * interval, accent: k % beatsPerBar === 0 })
      }
      return out
    },
  }
}
