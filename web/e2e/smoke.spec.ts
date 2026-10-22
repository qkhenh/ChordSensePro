import { expect, test } from '@playwright/test'

test('app boots and shows the product name', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 1, name: /chordsense pro/i })).toBeVisible()
})
