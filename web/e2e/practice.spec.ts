/// <reference lib="dom" />
import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'

// Needs the real backend + PostgreSQL: run with E2E_BACKEND=1 (the config starts uvicorn).
test.skip(!process.env.E2E_BACKEND, 'set E2E_BACKEND=1 to run against the real API')
test.describe.configure({ mode: 'serial', timeout: 90_000 })

interface Meta {
  beat_grid: { tempo_bpm: number; beats_per_bar: number; first_beat_offset_ms: number }
  duration_ms: number
}
const blues = JSON.parse(readFileSync('../contracts/fixtures/results/blues_f.result.json', 'utf8')).metadata as Meta
const BAR_MS = (blues.beat_grid.beats_per_bar * 60_000) / blues.beat_grid.tempo_bpm
const barStart = (k: number) => Math.round(blues.beat_grid.first_beat_offset_ms + k * BAR_MS)
const barEnd = (k: number) => Math.min(barStart(k + 1), blues.duration_ms)

declare global {
  interface Window {
    __audios: HTMLAudioElement[]
    __frames: [number, number][]
  }
}

let counter = 0

/** The role is applied to the sign-up answer only: Pro features are gated in the page (D3). */
async function signUp(page: Page, role: 'free' | 'paid' = 'free') {
  await page.addInitScript(() => {
    const Original = window.Audio
    window.__audios = []
    window.Audio = function (...args: ConstructorParameters<typeof Audio>) {
      const element = new Original(...args)
      window.__audios.push(element)
      return element
    } as unknown as typeof Audio
    window.Audio.prototype = Original.prototype
  })
  if (role !== 'free') {
    await page.route('**/api/v1/auth/register', async (route) => {
      const response = await route.fetch()
      const body = await response.json()
      body.user.role = role
      await route.fulfill({ response, json: body })
    })
  }
  await page.goto('/register')
  await page.getByLabel('Email').fill(`e2e-practice-${Date.now()}-${counter++}@example.com`)
  await page.getByLabel('Mật khẩu').fill('correct horse battery')
  await page.getByRole('button', { name: 'Đăng ký' }).click()
  await expect(page).toHaveURL(/\/workspace$/)
}

async function openDemo(page: Page, title: RegExp) {
  await page.getByRole('button', { name: title }).click()
  await expect(page).toHaveURL(/\/charts\/[0-9a-f-]{36}$/)
  await expect(page.getByRole('group', { name: 'Lưới hợp âm' })).toBeVisible()
  // the transport is ready once Play can be pressed
  await expect(page.getByRole('button', { name: 'Phát' })).toBeEnabled({ timeout: 20_000 })
}

const bar = (page: Page, index: number) => page.locator(`[data-bar="${index}"]`)
const lit = (page: Page) => page.locator('[data-current="true"]')
const audioMs = (page: Page) => page.evaluate(() => window.__audios[0].currentTime * 1000)

test('playing lights the bars one after the other', async ({ page }) => {
  await signUp(page)
  await openDemo(page, /ii–V–I/)
  await page.getByRole('button', { name: 'Phát' }).click()
  await expect(lit(page)).toHaveCount(1)
  const first = await lit(page).getAttribute('data-bar')
  await expect.poll(() => lit(page).getAttribute('data-bar'), { timeout: 8000 }).not.toBe(first)
})

test('clicking a bar jumps to its start and lights it', async ({ page }) => {
  await signUp(page)
  await openDemo(page, /12-bar blues/)
  await bar(page, 8).click()
  expect(Math.abs((await audioMs(page)) - barStart(8))).toBeLessThanOrEqual(100)
  await expect(bar(page, 8)).toHaveAttribute('data-current', 'true')
})

test('a loop of bars 9-12 plays on and returns to bar 9 within 100 ms of reaching its end', async ({ page }) => {
  await signUp(page)
  await openDemo(page, /12-bar blues/)
  await bar(page, 8).dragTo(bar(page, 11))
  await expect(page.getByText(/ô nhịp 9–12/i)).toBeVisible()
  await bar(page, 11).click() // into the last bar of the loop, a couple of seconds before its end
  await page.evaluate(() => {
    const audio = window.__audios[0]
    window.__frames = []
    const record = () => {
      window.__frames.push([performance.now(), audio.currentTime * 1000])
      requestAnimationFrame(record)
    }
    record()
  })
  await page.getByRole('button', { name: 'Phát' }).click()
  const end = barEnd(11)
  const overshoot = await page.evaluate(
    ([loopEnd]) =>
      new Promise<number>((resolve, reject) => {
        const started = performance.now()
        const look = () => {
          const frames = window.__frames
          for (let i = 1; i < frames.length; i++) {
            if (frames[i][1] < frames[i - 1][1] - 1000) {
              resolve(Math.max(0, frames[i - 1][1] - loopEnd)) // furthest the playhead was seen past B
              return
            }
          }
          if (performance.now() - started > 15_000) reject(new Error('the loop never wrapped'))
          else setTimeout(look, 50)
        }
        look()
      }),
    [end],
  )
  console.log(`[NFR-04] loop wrap: playhead seen ${overshoot.toFixed(1)} ms past B`) // eslint-disable-line no-console
  expect(overshoot).toBeLessThanOrEqual(100)
  expect(Math.abs((await audioMs(page)) - barStart(8))).toBeLessThan(3000) // playing on from bar 9
})

test('0.75x keeps the pitch for a Pro user', async ({ page }) => {
  await signUp(page, 'paid')
  await openDemo(page, /12-bar blues/)
  await page.getByRole('button', { name: '0.75×' }).click()
  expect(await page.evaluate(() => window.__audios[0].playbackRate)).toBe(0.75)
  expect(await page.evaluate(() => window.__audios[0].preservesPitch)).toBe(true)
})

test('a free user is offered the upgrade at 0.75x and stays at normal speed', async ({ page }) => {
  await signUp(page)
  await openDemo(page, /12-bar blues/)
  await page.getByRole('button', { name: '0.75×' }).click()
  await expect(page.getByRole('link', { name: 'Nâng cấp lên Pro' })).toBeVisible()
  expect(await page.evaluate(() => window.__audios[0].playbackRate)).toBe(1)
})

test('Basic and Precise change the chord names', async ({ page }) => {
  await signUp(page)
  await openDemo(page, /12-bar blues/)
  const names = () =>
    page.locator('[data-bar]').evaluateAll((cells) => cells.map((c) => c.getAttribute('aria-label')).join('|'))
  const precise = await names()
  await page.getByRole('button', { name: 'Cơ bản' }).click()
  const basic = await names()
  expect(basic).not.toBe(precise)
  await page.getByRole('button', { name: 'Chi tiết' }).click()
  expect(await names()).toBe(precise)
})

test('Space plays and pauses, the arrows move by a bar', async ({ page }) => {
  await signUp(page)
  await openDemo(page, /12-bar blues/)
  await page.locator('body').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('Space')
  await expect.poll(() => page.evaluate(() => window.__audios[0].paused)).toBe(false)
  await expect(page.getByRole('button', { name: 'Tạm dừng' })).toBeVisible() // the page has caught up with the player
  await page.keyboard.press('Space')
  await expect.poll(() => page.evaluate(() => window.__audios[0].paused)).toBe(true)
  await bar(page, 3).click()
  await page.locator('body').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('ArrowRight')
  await expect(bar(page, 4)).toHaveAttribute('data-current', 'true')
  await page.keyboard.press('ArrowLeft')
  await expect(bar(page, 3)).toHaveAttribute('data-current', 'true')
})

test('a YouTube chart whose player cannot load offers an upload instead', async ({ page }) => {
  await signUp(page)
  await page.route('https://www.youtube.com/**', (route) => route.abort())
  await page.route(/\/api\/v1\/charts\/[0-9a-f-]{36}$/, async (route) => {
    const response = await route.fetch()
    const chart = await response.json()
    chart.source = { ...chart.source, kind: 'url', playback: { kind: 'youtube', embed_id: 'dQw4w9WgXcQ', audio_url: null } }
    await route.fulfill({ response, json: chart })
  })
  await page.getByRole('button', { name: /12-bar blues/ }).click()
  await expect(page.getByRole('group', { name: 'Lưới hợp âm' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Tải tệp lên thay thế' })).toBeVisible({ timeout: 20_000 })
})

// Where the song is picked up again after a renewal is covered by the transport's unit tests: a
// browser cannot be made to lose its place here, so this checks the part only a real one shows.
test('an audio ticket that is refused is renewed and the song still plays', async ({ page }) => {
  await signUp(page)
  let ticketRequests = 0
  page.on('request', (request) => {
    if (request.method() === 'POST' && request.url().includes('/audio-ticket')) ticketRequests++
  })
  await openDemo(page, /12-bar blues/)
  expect(ticketRequests).toBe(1)
  let denied = 0
  await page.route(/\/charts\/[0-9a-f-]{36}\/audio\?ticket=/, (route) => {
    if (denied++ === 0) return route.fulfill({ status: 401, json: { code: 'invalid_ticket', message: 'expired' } })
    return route.continue()
  })
  await page.evaluate(() => {
    const audio = window.__audios[0]
    const { src } = audio
    audio.src = src // load again with the old ticket: the server refuses it once
  })
  await expect.poll(() => ticketRequests, { timeout: 15_000 }).toBe(2)
  await expect(page.getByRole('button', { name: 'Phát' })).toBeEnabled({ timeout: 15_000 })
  await expect(page.getByText(/không phát được bài này/i)).toHaveCount(0)
  await page.getByRole('button', { name: 'Phát' }).click()
  await expect.poll(() => page.evaluate(() => window.__audios[0].paused)).toBe(false)
})
