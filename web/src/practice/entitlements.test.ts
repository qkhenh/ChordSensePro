import { describe, expect, it } from 'vitest'
import type { Role } from '../api/types'
import { entitlementsFor } from './entitlements'

const ALL = { speed: true, songSyncMetronome: true, countIn: true, loopSpeed: true }
const NONE = { speed: false, songSyncMetronome: false, countIn: false, loopSpeed: false }

describe('entitlementsFor', () => {
  it.each<[Role, typeof ALL]>([
    ['guest', NONE],
    ['free', NONE],
    ['paid', ALL],
    ['teacher', ALL],
    ['student', ALL],
    ['expert', NONE],
    ['music_pm', NONE],
    ['it_admin', NONE],
  ])('%s', (role, expected) => {
    expect(entitlementsFor(role)).toEqual(expected)
  })
})
