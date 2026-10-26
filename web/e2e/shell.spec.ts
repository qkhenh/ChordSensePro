import { expect, test } from '@playwright/test'

// Needs the real backend + PostgreSQL: run with E2E_BACKEND=1 (the config then starts uvicorn).
test.skip(!process.env.E2E_BACKEND, 'set E2E_BACKEND=1 to run against the real API')

test('register, land in the shell with the quota badge, switch language and sign out', async ({ page }) => {
  const email = `e2e-${Date.now()}@example.com`

  await page.goto('/register')
  await page.getByLabel('Email').fill(email)
  await page.getByLabel('Mật khẩu').fill('correct horse battery')
  await page.getByRole('button', { name: 'Đăng ký' }).click()

  await expect(page).toHaveURL(/\/workspace$/)
  await expect(page.getByRole('navigation', { name: 'Điều hướng chính' })).toBeVisible()
  await expect(page.getByTestId('quota-badge')).toHaveText('Free · còn 5/5')

  await page.getByRole('button', { name: 'English' }).click()
  await expect(page.getByTestId('quota-badge')).toHaveText('Free · 5/5 left')

  await page.reload() // session restored from the refresh token, language from storage
  await expect(page.getByTestId('quota-badge')).toHaveText('Free · 5/5 left')

  await page.getByRole('button', { name: 'Sign out' }).click()
  await expect(page).toHaveURL(/\/login$/)
})
