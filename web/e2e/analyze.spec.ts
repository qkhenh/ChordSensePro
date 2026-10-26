import { randomBytes } from 'node:crypto'
import { expect, test, type Page } from '@playwright/test'

// Needs the real backend + PostgreSQL + ffmpeg: run with E2E_BACKEND=1 (the config starts uvicorn).
test.skip(!process.env.E2E_BACKEND, 'set E2E_BACKEND=1 to run against the real API')
test.describe.configure({ mode: 'serial', timeout: 90_000 })

let counter = 0

/**
 * A short mono WAV of random noise: the content cache is shared by every user and every earlier
 * run, so any repeated sound would be served from it (free) and skew the quota counts.
 */
function wav(seconds = 1): Buffer {
  const rate = 8000
  const data = randomBytes(rate * seconds * 2)
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + data.length, 4)
  header.write('WAVEfmt ', 8)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(rate, 24)
  header.writeUInt32LE(rate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(data.length, 40)
  return Buffer.concat([header, data])
}

async function signUp(page: Page) {
  await page.goto('/register')
  await page.getByLabel('Email').fill(`e2e-${Date.now()}-${counter++}@example.com`)
  await page.getByLabel('Mật khẩu').fill('correct horse battery')
  await page.getByRole('button', { name: 'Đăng ký' }).click()
  await expect(page).toHaveURL(/\/workspace$/)
}

async function analyzeFile(page: Page, name: string, buffer: Buffer = wav(), mimeType = 'audio/wav') {
  await page.getByLabel(/chọn tệp/i).setInputFiles({ name, mimeType, buffer })
  await page.getByRole('button', { name: 'Phân tích bài hát' }).click()
}

const badge = (page: Page) => page.getByTestId('quota-badge')

test('upload a WAV, watch the steps, land on the finished chart', async ({ page }) => {
  await signUp(page)
  await analyzeFile(page, 'E2E Song.wav')
  await expect(page).toHaveURL(/\/charts\/[0-9a-f-]{36}$/)
  await expect(page.getByRole('heading', { level: 2, name: 'E2E Song' })).toBeVisible()
  await expect(page.getByRole('row').nth(1)).toContainText(/:/) // at least one chord with a time
  await expect(badge(page)).toContainText('còn 4/5')
})

test('a text file with an audio name is refused with a clear reason and costs nothing', async ({ page }) => {
  await signUp(page)
  await analyzeFile(page, 'not-music.mp3', Buffer.from('this is plain text'), 'audio/mpeg')
  await expect(page.getByRole('alert')).toContainText('không phải âm thanh hợp lệ')
  await expect(page.getByRole('tab', { name: 'Tải lên' })).toBeVisible()
  await expect(badge(page)).toContainText('còn 5/5')
})

test('unsafe links are explained, a public link is analysed', async ({ page }) => {
  await signUp(page)
  await page.getByRole('tab', { name: 'Liên kết' }).click()
  const box = page.getByLabel(/dán liên kết/i)

  await box.fill('http://127.0.0.1/x.mp3')
  await page.getByRole('button', { name: 'Phân tích bài hát' }).click()
  await expect(page.getByRole('alert')).toContainText('địa chỉ nội bộ')
  await page.getByRole('button', { name: 'Tải tệp lên thay thế' }).click()
  await expect(page.getByRole('tab', { name: 'Tải lên' })).toHaveAttribute('aria-selected', 'true')

  await page.getByRole('tab', { name: 'Liên kết' }).click()
  await box.fill('http://93.184.216.34/Public%20Song.mp3') // an IP literal: no DNS, never fetched
  await page.getByRole('button', { name: 'Phân tích bài hát' }).click()
  await expect(page).toHaveURL(/\/charts\/[0-9a-f-]{36}$/)
  await expect(page.getByRole('heading', { level: 2, name: 'Public Song' })).toBeVisible()
})

test('a failing analysis says why and can be retried', async ({ page }) => {
  await signUp(page)
  await analyzeFile(page, 'fail-ai.wav')
  await expect(page.getByRole('alert')).toContainText('không phân tích được')
  await expect(page.getByRole('button', { name: 'Thử lại' })).toBeVisible()
  await expect(badge(page)).toContainText('còn 5/5') // the charge came back
})

test('reloading the page in the middle of a job keeps following it', async ({ page }) => {
  await signUp(page)
  await analyzeFile(page, 'slow-song.wav')
  await expect(page.getByRole('progressbar')).toBeVisible()
  await page.reload()
  await expect(page.getByRole('progressbar')).toBeVisible()
  await expect(page).toHaveURL(/\/charts\/[0-9a-f-]{36}$/, { timeout: 30_000 })
})

test('a demo song opens its chart without spending the allowance', async ({ page }) => {
  await signUp(page)
  await expect(badge(page)).toContainText('còn 5/5')
  await page.getByRole('button', { name: /12-bar blues/ }).click()
  await expect(page).toHaveURL(/\/charts\/[0-9a-f-]{36}$/)
  await expect(page.getByRole('heading', { level: 2, name: /12-bar blues/ })).toBeVisible()
  await expect(badge(page)).toContainText('còn 5/5')
})

test('the sixth analysis of the day is refused with the limit and the reset time', async ({ page }) => {
  await signUp(page)
  for (let i = 0; i < 5; i++) {
    await page.goto('/workspace')
    await analyzeFile(page, `daily-${i}.wav`)
    await expect(page).toHaveURL(/\/charts\//)
  }
  await page.goto('/workspace')
  await analyzeFile(page, 'one-too-many.wav')
  await expect(page.getByRole('alert')).toContainText('Bạn đã dùng hết 5 lượt phân tích hôm nay')
  await expect(page.getByRole('link', { name: 'Nâng cấp Pro' })).toBeVisible()
  await expect(badge(page)).toContainText('hết lượt')
})
