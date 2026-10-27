import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { describe, expect, it } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import { buildRoutes } from '../routes'
import { apiError, authResponse, http, HttpResponse, quotaOut, server } from '../test/server'

const API = '*/api/v1'

function renderShell(path = '/') {
  server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage('refresh-stored') })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: [path] })
  render(<RouterProvider router={router} />)
  return { router, user: userEvent.setup() }
}

const nav = () => screen.findByRole('navigation', { name: 'Điều hướng chính' })

describe('workspace shell', () => {
  it('sends / to the workspace', async () => {
    const { router } = renderShell('/')
    await nav()
    await waitFor(() => expect(router.state.location.pathname).toBe('/workspace'))
  })

  it('lists Library, Workspace and Dictionary and marks the current one', async () => {
    renderShell('/workspace')
    const sidebar = await nav()
    expect(within(sidebar).getAllByRole('link').map((l) => l.textContent)).toEqual([
      'Thư viện',
      'Workspace',
      'Từ điển hợp âm',
    ])
    expect(within(sidebar).getByRole('link', { name: 'Workspace' })).toHaveAttribute('aria-current', 'page')
  })

  it.each([
    ['Thư viện', '/library'],
    ['Từ điển hợp âm', '/dictionary'],
  ])('opens %s from the sidebar', async (label, path) => {
    const { router, user } = renderShell('/workspace')
    await user.click(within(await nav()).getByRole('link', { name: label }))
    expect(router.state.location.pathname).toBe(path)
    expect(await screen.findByRole('heading', { level: 2, name: label })).toBeInTheDocument()
  })

  it('shows a skeleton placeholder on the still-empty pages', async () => {
    renderShell('/dictionary')
    expect(await screen.findByTestId('page-skeleton')).toBeInTheDocument()
  })

  it('shows who is signed in and signs out from the header', async () => {
    const { router, user } = renderShell('/workspace')
    const header = await screen.findByRole('banner')
    expect(within(header).getByText('ann@example.com')).toBeInTheDocument()
    server.use(http.post(`${API}/auth/logout`, () => new HttpResponse(null, { status: 204 })))
    await user.click(within(header).getByRole('button', { name: 'Đăng xuất' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/login'))
  })
})

describe('quota badge', () => {
  const badge = () => screen.findByTestId('quota-badge')

  it('shows the tier and how many analyses are left today', async () => {
    renderShell()
    await waitFor(async () => expect(await badge()).toHaveTextContent('Free · còn 3/5'))
  })

  it('shows a skeleton while the quota is loading', async () => {
    server.use(
      http.get(`${API}/users/me/quota`, async () => {
        await new Promise((r) => setTimeout(r, 50))
        return HttpResponse.json(quotaOut())
      }),
    )
    renderShell()
    expect(await badge()).toHaveAttribute('aria-busy', 'true')
    await waitFor(() => expect(screen.getByTestId('quota-badge')).toHaveTextContent('còn 3/5'))
  })

  it('says unlimited when the tier has no daily limit', async () => {
    server.use(
      http.get(`${API}/users/me/quota`, () =>
        HttpResponse.json(quotaOut({ tier: 'pro', limit: null, remaining: null, used: 40 })),
      ),
    )
    renderShell()
    await waitFor(async () => expect(await badge()).toHaveTextContent('Pro · không giới hạn'))
  })

  it('flags an exhausted quota', async () => {
    server.use(http.get(`${API}/users/me/quota`, () => HttpResponse.json(quotaOut({ used: 5, remaining: 0 }))))
    renderShell()
    await waitFor(async () => expect(await badge()).toHaveTextContent('Free · hết lượt hôm nay'))
    expect(await badge()).toHaveAttribute('data-exhausted', 'true')
  })

  it('tells the user when the quota cannot be loaded, without hiding the shell', async () => {
    server.use(http.get(`${API}/users/me/quota`, () => apiError(500, 'internal_error')))
    renderShell()
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(await badge()).toHaveTextContent('—')
    expect(await nav()).toBeInTheDocument()
  })
})

describe('language', () => {
  it('switches the shell to English and remembers the choice', async () => {
    const { user } = renderShell('/workspace')
    await nav()
    await user.click(screen.getByRole('button', { name: 'English' }))
    expect(await screen.findByRole('navigation', { name: 'Main navigation' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Library' })).toBeInTheDocument()
    await waitFor(async () =>
      expect(await screen.findByTestId('quota-badge')).toHaveTextContent('Free · 3/5 left'),
    )
    expect(localStorage.getItem('chordsense.lang')).toBe('en')
  })

  it('starts in the remembered language', async () => {
    localStorage.setItem('chordsense.lang', 'en')
    renderShell('/workspace')
    expect(await screen.findByRole('link', { name: 'Dictionary' })).toBeInTheDocument()
    expect(document.documentElement.lang).toBe('en')
  })

  it('ignores an unknown stored language', async () => {
    localStorage.setItem('chordsense.lang', 'xx')
    renderShell('/workspace')
    expect(await nav()).toBeInTheDocument()
    expect(document.documentElement.lang).toBe('vi')
  })
})
