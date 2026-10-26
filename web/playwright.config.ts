import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  reporter: 'list',
  use: { baseURL: 'http://localhost:5173', trace: 'on-first-retry' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    {
      // Firefox only runs the timing specs (NFR-01/04 ask for both browsers).
      name: 'firefox',
      testMatch: /.*\.perf\.spec\.ts/,
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: { firefoxUserPrefs: { 'media.autoplay.default': 0, 'media.autoplay.blocking_policy': 0 } },
      },
    },
  ],
  webServer: [
    {
      command: 'npm run dev -- --port 5173 --strictPort',
      url: 'http://localhost:5173',
      reuseExistingServer: true,
    },
    // Real API for e2e/shell.spec.ts (needs PostgreSQL as configured in DE/.env).
    ...(process.env.E2E_BACKEND
      ? [
          {
            // Demo audio is rendered once (kept if present); then the API on IPv4 loopback.
            command:
              'uv run python scripts/gen_demo_audio.py && uv run uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000',
            cwd: '../DE',
            url: 'http://127.0.0.1:8000/healthz',
            timeout: 180_000,
            reuseExistingServer: true,
          },
        ]
      : []),
  ],
})
