import { beforeEach, describe, expect, it, vi } from 'vitest'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'
import { ApiError, createApiClient, memoryStorage } from './http'

const API = '*/api/v1'

function newClient() {
  const storage = memoryStorage()
  return { storage, client: createApiClient({ storage }) }
}

describe('api client', () => {
  beforeEach(() => {
    server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse({ access_token: 'access-old' }))))
  })

  it('logs in, keeps the refresh token in storage and sends the access token', async () => {
    const { client, storage } = newClient()
    const seen: (string | null)[] = []
    server.use(
      http.get(`${API}/users/me`, ({ request }) => {
        seen.push(request.headers.get('authorization'))
        return HttpResponse.json({ ok: true })
      }),
    )

    const res = await client.login('ann@example.com', 'secret-pass')
    expect(res.user.email).toBe('ann@example.com')
    expect(storage.getRefresh()).toBe('refresh-1')

    await client.request('/users/me')
    expect(seen).toEqual(['Bearer access-old'])
  })

  it('does not send credentials on unauthenticated calls', async () => {
    const { client } = newClient()
    let header: string | null = 'unset'
    server.use(
      http.get(`${API}/public`, ({ request }) => {
        header = request.headers.get('authorization')
        return HttpResponse.json({})
      }),
    )
    await client.login('ann@example.com', 'secret-pass')
    await client.request('/public', { auth: false })
    expect(header).toBeNull()
  })

  it('refreshes once on token_expired, retries the request and stores the rotated token', async () => {
    const { client, storage } = newClient()
    await client.login('ann@example.com', 'secret-pass')
    server.use(
      http.get(`${API}/users/me`, ({ request }) =>
        request.headers.get('authorization') === 'Bearer access-new'
          ? HttpResponse.json({ email: 'ann@example.com' })
          : apiError(401, 'token_expired'),
      ),
      http.post(`${API}/auth/refresh`, () =>
        HttpResponse.json(authResponse({ access_token: 'access-new', refresh_token: 'refresh-2' })),
      ),
    )

    await expect(client.request('/users/me')).resolves.toEqual({ email: 'ann@example.com' })
    expect(storage.getRefresh()).toBe('refresh-2')
  })

  it('shares a single refresh between concurrent requests', async () => {
    const { client } = newClient()
    await client.login('ann@example.com', 'secret-pass')
    const refresh = vi.fn()
    server.use(
      http.get(`${API}/users/me`, ({ request }) =>
        request.headers.get('authorization') === 'Bearer access-new'
          ? HttpResponse.json({ ok: true })
          : apiError(401, 'token_expired'),
      ),
      http.post(`${API}/auth/refresh`, async () => {
        refresh()
        await new Promise((r) => setTimeout(r, 20))
        return HttpResponse.json(authResponse({ access_token: 'access-new', refresh_token: 'refresh-2' }))
      }),
    )

    await Promise.all([client.request('/users/me'), client.request('/users/me'), client.request('/users/me')])
    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('clears the session and notifies listeners when the refresh token is rejected', async () => {
    const { client, storage } = newClient()
    await client.login('ann@example.com', 'secret-pass')
    const expired = vi.fn()
    client.onSessionExpired(expired)
    server.use(
      http.get(`${API}/users/me`, () => apiError(401, 'token_expired')),
      http.post(`${API}/auth/refresh`, () => apiError(401, 'token_revoked')),
    )

    await expect(client.request('/users/me')).rejects.toMatchObject({ status: 401 })
    expect(storage.getRefresh()).toBeNull()
    expect(expired).toHaveBeenCalledTimes(1)
  })

  it('gives up after one retry instead of looping on a persistent 401', async () => {
    const { client } = newClient()
    await client.login('ann@example.com', 'secret-pass')
    const refresh = vi.fn()
    server.use(
      http.get(`${API}/users/me`, () => apiError(401, 'invalid_token')),
      http.post(`${API}/auth/refresh`, () => {
        refresh()
        return HttpResponse.json(authResponse({ access_token: 'access-new', refresh_token: 'refresh-2' }))
      }),
    )

    await expect(client.request('/users/me')).rejects.toMatchObject({ status: 401, code: 'invalid_token' })
    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('does not try to refresh when there is no refresh token', async () => {
    const { client } = newClient()
    server.use(http.get(`${API}/users/me`, () => apiError(401, 'not_authenticated')))
    await expect(client.request('/users/me')).rejects.toMatchObject({ status: 401, code: 'not_authenticated' })
  })

  it('surfaces the server error code and message', async () => {
    const { client } = newClient()
    server.use(http.post(`${API}/auth/register`, () => apiError(409, 'email_taken', 'taken')))
    const error = await client.register('ann@example.com', 'secret-pass').catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 409, code: 'email_taken', message: 'taken' })
  })

  it('keeps field-level validation detail and the request id on the error', async () => {
    const { client } = newClient()
    server.use(
      http.post(`${API}/auth/register`, () =>
        HttpResponse.json(
          {
            code: 'validation_error',
            message: 'Request validation failed',
            detail: [{ field: 'email', message: 'not an email' }],
          },
          { status: 422, headers: { 'X-Request-ID': 'req-42' } },
        ),
      ),
    )
    const error = await client.register('bad', 'secret-pass').catch((e) => e)
    expect(error).toMatchObject({
      code: 'validation_error',
      requestId: 'req-42',
      fieldErrors: [{ field: 'email', message: 'not an email' }],
    })
  })

  it('has no field errors or request id when the server sent none', async () => {
    const { client } = newClient()
    server.use(http.get(`${API}/users/me`, () => apiError(403, 'forbidden')))
    const error = (await client.request('/users/me').catch((e) => e)) as ApiError
    expect(error.fieldErrors).toEqual([])
    expect(error.requestId).toBeNull()
  })

  it('maps a network failure to a network_error ApiError', async () => {
    const { client } = newClient()
    server.use(http.get(`${API}/users/me`, () => HttpResponse.error()))
    await expect(client.request('/users/me')).rejects.toMatchObject({ status: 0, code: 'network_error' })
  })

  it('copes with an error body that is not JSON', async () => {
    const { client } = newClient()
    server.use(http.get(`${API}/users/me`, () => new HttpResponse('Bad gateway', { status: 502 })))
    await expect(client.request('/users/me')).rejects.toMatchObject({ status: 502, code: 'http_502' })
  })

  describe('restore', () => {
    it('returns null without calling the server when nothing is stored', async () => {
      const { client } = newClient()
      await expect(client.restore()).resolves.toBeNull()
    })

    it('exchanges a stored refresh token for a fresh session', async () => {
      const storage = memoryStorage('refresh-stored')
      const client = createApiClient({ storage })
      server.use(
        http.post(`${API}/auth/refresh`, async ({ request }) => {
          expect(await request.json()).toEqual({ refresh_token: 'refresh-stored' })
          return HttpResponse.json(authResponse({ refresh_token: 'refresh-rotated' }))
        }),
      )
      const res = await client.restore()
      expect(res?.user.email).toBe('ann@example.com')
      expect(storage.getRefresh()).toBe('refresh-rotated')
    })

    it('returns null and forgets the token when it is no longer valid', async () => {
      const storage = memoryStorage('refresh-dead')
      const client = createApiClient({ storage })
      server.use(http.post(`${API}/auth/refresh`, () => apiError(401, 'token_revoked')))
      await expect(client.restore()).resolves.toBeNull()
      expect(storage.getRefresh()).toBeNull()
    })
  })

  describe('logout', () => {
    it('forgets the session locally and revokes the refresh token on the server', async () => {
      const { client, storage } = newClient()
      await client.login('ann@example.com', 'secret-pass')
      let revoked: unknown
      server.use(
        http.post(`${API}/auth/logout`, async ({ request }) => {
          revoked = await request.json()
          return new HttpResponse(null, { status: 204 })
        }),
      )

      await client.logout()
      expect(revoked).toEqual({ refresh_token: 'refresh-1' })
      expect(storage.getRefresh()).toBeNull()
      expect(client.getAccessToken()).toBeNull()
    })

    it('still signs out locally when the server cannot be reached', async () => {
      const { client, storage } = newClient()
      await client.login('ann@example.com', 'secret-pass')
      server.use(http.post(`${API}/auth/logout`, () => HttpResponse.error()))

      await client.logout()
      expect(storage.getRefresh()).toBeNull()
      expect(client.getAccessToken()).toBeNull()
    })

    it('does not call the server when nobody is signed in', async () => {
      const { client } = newClient() // no handler registered: a request would fail the frame
      await expect(client.logout()).resolves.toBeUndefined()
    })
  })
})
