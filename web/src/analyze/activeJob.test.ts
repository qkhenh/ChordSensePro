import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearActiveJob, readActiveJob, writeActiveJob } from './activeJob'

afterEach(() => {
  sessionStorage.clear()
  vi.restoreAllMocks()
})

describe('active job memory', () => {
  it('remembers the job being watched across a page reload', () => {
    expect(readActiveJob()).toBeNull()
    writeActiveJob('job-1')
    expect(readActiveJob()).toBe('job-1')
  })

  it('forgets it when asked', () => {
    writeActiveJob('job-1')
    clearActiveJob()
    expect(readActiveJob()).toBeNull()
  })

  it('works without storage (private mode, blocked site data)', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(readActiveJob()).toBeNull()
    expect(() => writeActiveJob('job-1')).not.toThrow()
    expect(() => clearActiveJob()).not.toThrow()
  })
})
