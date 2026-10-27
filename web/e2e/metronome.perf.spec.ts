/// <reference lib="dom" />
import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'

/**
 * NFR-04: a metronome click lands within 10 ms of its beat on the grid. The real click and beat
 * code is rendered by an OfflineAudioContext in the browser (so it needs no sound card and has
 * no scheduling jitter of its own), the onsets are found in the samples and compared with the
 * beats of the 12-bar blues grid. Runs in Chromium and Firefox.
 */
const result = JSON.parse(readFileSync('../contracts/fixtures/results/blues_f.result.json', 'utf8'))

test('32 clicks fall within 10 ms of the grid beats, and the first beat of a bar is louder', async ({
  page,
  browserName,
}) => {
  await page.goto('/login') // any page of the dev server: it only serves the modules
  const report = await page.evaluate(async (metadata) => {
    const { buildGrid } = await import(/* @vite-ignore */ '/src/grid/' + 'grid.ts')
    const { songBeats } = await import(/* @vite-ignore */ '/src/metronome/' + 'beats.ts')
    const { scheduleClick } = await import(/* @vite-ignore */ '/src/metronome/' + 'click.ts')

    const grid = buildGrid(metadata, [])
    const beats = songBeats(grid).beatsBetween(0, metadata.duration_ms).slice(0, 32)
    const rate = 44100
    const seconds = beats[beats.length - 1].ms / 1000 + 0.5
    const ctx = new OfflineAudioContext(1, Math.ceil(seconds * rate), rate)
    for (const beat of beats) {
      scheduleClick(ctx, beat.ms / 1000, { accent: beat.accent, timbre: 'woodblock', volume: 0.8 })
    }
    const samples = (await ctx.startRendering()).getChannelData(0)

    // An onset is the first sample above the threshold after at least 100 ms of quiet.
    const threshold = 0.02
    const quiet = Math.round(0.1 * rate)
    const onsets: { ms: number; peak: number }[] = []
    let lastLoud = -quiet
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) < threshold) continue
      if (i - lastLoud >= quiet) onsets.push({ ms: (i / rate) * 1000, peak: 0 })
      lastLoud = i
      onsets[onsets.length - 1].peak = Math.max(onsets[onsets.length - 1].peak, Math.abs(samples[i]))
    }
    return { beats, onsets }
  }, result.metadata)

  expect(report.onsets).toHaveLength(report.beats.length)
  const errors = report.onsets.map((onset, i) => Math.abs(onset.ms - report.beats[i].ms))
  const worst = Math.max(...errors)
  console.log(`[NFR-04] ${browserName}: ${report.beats.length} clicks, worst offset ${worst.toFixed(2)} ms`) // eslint-disable-line no-console
  expect(worst).toBeLessThanOrEqual(10)

  const accented = report.onsets.filter((_, i) => report.beats[i].accent).map((o) => o.peak)
  const normal = report.onsets.filter((_, i) => !report.beats[i].accent).map((o) => o.peak)
  expect(Math.min(...accented)).toBeGreaterThan(Math.max(...normal))
})
