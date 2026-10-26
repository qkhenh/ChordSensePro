import { expect, test } from '@playwright/test'

/**
 * NFR-01: the lit bar stays within 100 ms of the bar the audio clock says is playing (p95), and
 * the light changes no more than 100 ms (p95) after the song enters a new bar, at about 60 fps, in Chromium and Firefox. Each run plays a generated click track through the
 * real grid + playhead + HTML audio transport (route /__spike/sync, dev server only).
 *
 * SYNC_SECONDS (default 20) sets the length of a run; the formal check is SYNC_SECONDS=240.
 * SYNC_RUNS (default 3) is the number of runs.
 */
interface Spike {
  __syncStats?: {
    samples: number
    p50: number
    p95: number
    max: number
    meanFps: number
    flips: number
    lagP95: number
    lagMax: number
  }
}

const SECONDS = Number(process.env.SYNC_SECONDS ?? 20)
const RUNS = Number(process.env.SYNC_RUNS ?? 3)

test.describe.configure({ mode: 'serial' })

for (let run = 1; run <= RUNS; run++) {
  test(`run ${run}/${RUNS}: lit bar vs audio clock over ${SECONDS} s`, async ({ page, browserName }) => {
    test.setTimeout((SECONDS + 60) * 1000)
    await page.goto(`/__spike/sync?seconds=${SECONDS}`)
    await page.getByRole('button', { name: 'Start' }).click()
    await page.waitForFunction(() => (globalThis as unknown as Spike).__syncStats !== undefined, null, {
      timeout: (SECONDS + 40) * 1000,
    })
    const stats = await page.evaluate(() => (globalThis as unknown as Spike).__syncStats)
    console.log(`[NFR-01] ${browserName} run ${run}: ${JSON.stringify(stats)}`) // eslint-disable-line no-console
    test.info().annotations.push({ type: 'nfr-01', description: JSON.stringify(stats) })

    expect(stats?.samples ?? 0).toBeGreaterThan(SECONDS * 30)
    expect(stats?.p95 ?? Infinity).toBeLessThanOrEqual(100)
    expect(stats?.flips ?? 0).toBeGreaterThan(SECONDS / 2 - 3) // a bar every 2 s
    expect(stats?.lagP95 ?? Infinity).toBeLessThanOrEqual(100)
    expect(stats?.meanFps ?? 0).toBeGreaterThanOrEqual(50)
  })
}
