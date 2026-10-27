import type { Role } from '../api/types'

export interface Entitlements {
  /** Playing at 0.5× and 0.75×. */
  speed: boolean
  /** A metronome that follows the song's own beats. */
  songSyncMetronome: boolean
  countIn: boolean
  /** Looping a section at a reduced speed. */
  loopSpeed: boolean
}

export type Feature = 'speed' | 'songSync' | 'countIn'

// Same set as PRO_ROLES in DE/src/identity/domain/models.py (Education seats included).
const PRO: ReadonlySet<Role> = new Set<Role>(['paid', 'teacher', 'student'])

/**
 * What the practice workspace lets a role do. Free keeps the grid, a single loop at 1× and the
 * manual metronome; M3 only gates in the page, because no endpoint needs the check yet.
 */
export function entitlementsFor(role: Role): Entitlements {
  const pro = PRO.has(role)
  return { speed: pro, songSyncMetronome: pro, countIn: pro, loopSpeed: pro }
}
