import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { describe, expect, it } from 'vitest'
import { createApiClient, memoryStorage } from './api/http'
import { buildRoutes } from './routes'
import { apiError, authResponse, http, HttpResponse, server } from './test/server'

const API = '*/api/v1'

function renderAt(path: string, storedRefresh?: string) {
  const client = createApiClient({ storage: memoryStorage(storedRefresh) })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: [path] })
  render(<RouterProvider router={router} />)
  return { router, client, user: userEvent.setup() }
}

describe('route guard', () => {
  it('sends an anonymous visitor to the login page', async () => {
    const { router } = renderAt('/')
    expect(await screen.findByRole('heading', { name: 'Đăng nhập' })).toBeInTheDocument()
    expect(router.state.location.pathname).toBe('/login')
  })

  it('restores the session from a stored refresh token', async () => {
    server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
    renderAt('/', 'refresh-stored')
    expect(await screen.findByText('ann@example.com')).toBeInTheDocument()
  })

  it('falls back to the login page when the stored token is dead', async () => {
    server.use(http.post(`${API}/auth/refresh`, () => apiError(401, 'token_revoked')))
    renderAt('/', 'refresh-dead')
    expect(await screen.findByRole('heading', { name: 'Đăng nhập' })).toBeInTheDocument()
  })

  it('returns to the page the visitor originally asked for after logging in', async () => {
    server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse())))
    const { router, user } = renderAt('/workspace')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'secret-pass')
    await user.click(screen.getByRole('button', { name: 'Đăng nhập' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/workspace'))
  })

  it('keeps a signed-in user away from the login page', async () => {
    server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
    const { router } = renderAt('/login', 'refresh-stored')
    await waitFor(() => expect(router.state.location.pathname).toBe('/'))
  })

  it('logs out: forgets the session and returns to the login page', async () => {
    server.use(
      http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())),
      http.post(`${API}/auth/logout`, () => new HttpResponse(null, { status: 204 })),
    )
    const { router, user, client } = renderAt('/', 'refresh-stored')
    await user.click(await screen.findByRole('button', { name: 'Đăng xuất' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/login'))
    expect(client.getAccessToken()).toBeNull()
  })
})

describe('login page', () => {
  it('rejects a malformed email without calling the server', async () => {
    const { user } = renderAt('/login')
    await user.type(await screen.findByLabelText('Email'), 'not-an-email')
    await user.type(screen.getByLabelText('Mật khẩu'), 'secret-pass')
    await user.click(screen.getByRole('button', { name: 'Đăng nhập' }))
    expect(await screen.findByText('Email không hợp lệ')).toBeInTheDocument()
  })

  it('shows a friendly message for wrong credentials and stays on the page', async () => {
    server.use(http.post(`${API}/auth/login`, () => apiError(401, 'invalid_credentials')))
    const { router, user } = renderAt('/login')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'wrong-pass')
    await user.click(screen.getByRole('button', { name: 'Đăng nhập' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Email hoặc mật khẩu không đúng')
    expect(router.state.location.pathname).toBe('/login')
  })

  it('disables the button while the request is in flight', async () => {
    server.use(
      http.post(`${API}/auth/login`, async () => {
        await new Promise((r) => setTimeout(r, 50))
        return HttpResponse.json(authResponse())
      }),
    )
    const { user } = renderAt('/login')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'secret-pass')
    await user.click(screen.getByRole('button', { name: 'Đăng nhập' }))
    expect(screen.getByRole('button', { name: /đang/i })).toBeDisabled()
  })

  it('reports an unreachable server', async () => {
    server.use(http.post(`${API}/auth/login`, () => HttpResponse.error()))
    const { user } = renderAt('/login')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'secret-pass')
    await user.click(screen.getByRole('button', { name: 'Đăng nhập' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Không kết nối được máy chủ')
  })
})

describe('register page', () => {
  it('requires at least eight characters before calling the server', async () => {
    const { user } = renderAt('/register')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'short')
    await user.click(screen.getByRole('button', { name: 'Đăng ký' }))
    expect(await screen.findByText('Mật khẩu cần ít nhất 8 ký tự')).toBeInTheDocument()
  })

  it('creates the account, signs in and lands on the workspace', async () => {
    let body: unknown
    server.use(
      http.post(`${API}/auth/register`, async ({ request }) => {
        body = await request.json()
        return HttpResponse.json(authResponse(), { status: 201 })
      }),
    )
    const { router, user } = renderAt('/register')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'secret-pass')
    await user.type(screen.getByLabelText('Họ tên (tuỳ chọn)'), 'Ann')
    await user.click(screen.getByRole('button', { name: 'Đăng ký' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/'))
    expect(body).toEqual({ email: 'ann@example.com', password: 'secret-pass', name: 'Ann' })
  })

  it('tells the visitor when the email is already registered', async () => {
    server.use(http.post(`${API}/auth/register`, () => apiError(409, 'email_taken')))
    const { user } = renderAt('/register')
    await user.type(await screen.findByLabelText('Email'), 'ann@example.com')
    await user.type(screen.getByLabelText('Mật khẩu'), 'secret-pass')
    await user.click(screen.getByRole('button', { name: 'Đăng ký' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Email này đã được đăng ký')
  })

  it('links to the login page', async () => {
    const { user } = renderAt('/register')
    await user.click(await screen.findByRole('link', { name: 'Đăng nhập' }))
    expect(await screen.findByRole('heading', { name: 'Đăng nhập' })).toBeInTheDocument()
  })
})
