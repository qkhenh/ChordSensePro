import { describe, expect, it } from 'vitest'
import type { AnalysisResult, Chart, JobProgress } from './types'

// The same JSON the backend validates with Pydantic (contracts/fixtures).
const results = import.meta.glob<AnalysisResult>('../../../contracts/fixtures/results/*.json', {
  eager: true,
  import: 'default',
})
const charts = import.meta.glob<Chart>('../../../contracts/fixtures/charts/*.chart.json', {
  eager: true,
  import: 'default',
})
const errors = import.meta.glob<Record<string, JobProgress>>(
  '../../../contracts/fixtures/jobs/errors.json',
  { eager: true, import: 'default' },
)

describe('shared contract fixtures', () => {
  it('are discovered by the bundler', () => {
    expect(Object.keys(results).length).toBeGreaterThanOrEqual(7)
    expect(Object.keys(charts).length).toBeGreaterThanOrEqual(3)
  })

  it.each(Object.entries(results))('%s keeps the timeline invariants the grid relies on', (_, r) => {
    expect(r.schema_version).toBe('1.0')
    let previousEnd = 0
    for (const e of r.chord_events) {
      expect(e.end_ms).toBeGreaterThan(e.start_ms)
      expect(e.start_ms).toBeGreaterThanOrEqual(previousEnd)
      previousEnd = e.end_ms
    }
    expect(previousEnd).toBeLessThanOrEqual(r.metadata.duration_ms)
  })

  it('carry no Roman numerals: harmonic analysis is derived on the client', () => {
    for (const r of Object.values(results)) {
      for (const e of r.chord_events) expect(e).not.toHaveProperty('roman')
    }
  })

  it('model an embedded-player chart without a server audio URL', () => {
    const youtube = Object.values(charts).find((c) => c.source.playback.kind === 'youtube')
    expect(youtube?.source.playback.embed_id).toBeTruthy()
    expect(youtube?.source.playback.audio_url).toBeNull()
  })

  it('keep the AI baseline separate from user overrides', () => {
    const edited = Object.values(charts).find((c) => c.overrides.length > 0)
    expect(edited?.is_customized).toBe(true)
    expect(edited?.ai_events.every((e) => !e.is_user_edited)).toBe(true)
  })

  it('provide a failed job for every error code', () => {
    const [byCode] = Object.values(errors)
    expect(Object.keys(byCode).sort()).toEqual(
      [
        'ai_failure',
        'analysis_timeout',
        'duration_too_long',
        'file_too_large',
        'quota_exceeded',
        'ssrf_blocked',
        'unsupported_format',
        'url_unresolvable',
      ].sort(),
    )
  })
})
