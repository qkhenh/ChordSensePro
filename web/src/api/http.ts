import type { components } from './generated/schema'

export type AuthResponse = components['schemas']['AuthResponse']

export interface FieldError {
  field: string
  message: string
}

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  /** Per-field problems of a `validation_error`. */
  fieldErrors: FieldError[] = []
  /** `X-Request-ID` of the failed call, to quote when reporting a problem. */
  requestId: string | null = null

  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function parseFieldErrors(detail: unknown): FieldError[] {
  if (!Array.isArray(detail)) return []
  return detail.flatMap((d) =>
    d && typeof d.field === 'string' && typeof d.message === 'string' ? [{ field: d.field, message: d.message }] : [],
  )
}

/** Only the refresh token is persisted; the short-lived access token stays in memory. */
export interface TokenStorage {
  getRefresh(): string | null
  setRefresh(token: string | null): void
}

export function memoryStorage(initial: string | null = null): TokenStorage {
  let value = initial
  return {
    getRefresh: () => value,
    setRefresh: (token) => {
      value = token
    },
  }
}

export function localStorageStorage(key = 'chordsense.refresh'): TokenStorage {
  // localStorage can be missing or throw (private mode, blocked site data).
  return {
    getRefresh() {
      try {
        return localStorage.getItem(key)
      } catch {
        return null
      }
    },
    setRefresh(token) {
      try {
        if (token === null) localStorage.removeItem(key)
        else localStorage.setItem(key, token)
      } catch {
        /* session simply will not survive a reload */
      }
    },
  }
}

export interface RequestOptions {
  method?: string
  body?: unknown
  /** Attach the access token and refresh it on 401 (default true). */
  auth?: boolean
}

export interface ApiClientOptions {
  baseUrl?: string
  storage?: TokenStorage
}

// A 401 with one of these codes may be cured by exchanging the refresh token.
const REFRESHABLE = new Set(['token_expired', 'invalid_token', 'not_authenticated'])

export function createApiClient({ baseUrl = '/api/v1', storage = localStorageStorage() }: ApiClientOptions = {}) {
  let accessToken: string | null = null
  let inflightRefresh: Promise<AuthResponse> | null = null
  const expiredListeners = new Set<() => void>()

  const url = (path: string) => new URL(`${baseUrl}${path}`, globalThis.location?.origin ?? 'http://localhost')

  async function send(path: string, { method = 'GET', body }: RequestOptions, token: string | null) {
    const headers: Record<string, string> = { Accept: 'application/json' }
    if (body !== undefined) headers['Content-Type'] = 'application/json'
    if (token) headers.Authorization = `Bearer ${token}`
    try {
      return await fetch(url(path), {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      })
    } catch {
      throw new ApiError(0, 'network_error', 'Cannot reach the server')
    }
  }

  async function toApiError(res: Response): Promise<ApiError> {
    let error = new ApiError(res.status, `http_${res.status}`, res.statusText || `HTTP ${res.status}`)
    try {
      const data = await res.json()
      if (data && typeof data.code === 'string') {
        error = new ApiError(res.status, data.code, typeof data.message === 'string' ? data.message : data.code)
        error.fieldErrors = parseFieldErrors(data.detail)
      }
    } catch {
      /* body was not JSON */
    }
    error.requestId = res.headers.get('x-request-id')
    return error
  }

  function apply(data: AuthResponse): AuthResponse {
    accessToken = data.access_token
    storage.setRefresh(data.refresh_token)
    return data
  }

  function endSession() {
    accessToken = null
    storage.setRefresh(null)
  }

  async function doRefresh(): Promise<AuthResponse> {
    const token = storage.getRefresh()
    if (!token) throw new ApiError(401, 'not_authenticated', 'Not signed in')
    try {
      const res = await send('/auth/refresh', { method: 'POST', body: { refresh_token: token } }, null)
      if (!res.ok) throw await toApiError(res)
      return apply((await res.json()) as AuthResponse)
    } catch (error) {
      // Being offline must not sign the user out; a rejected token must.
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
        endSession()
        expiredListeners.forEach((listener) => listener())
      }
      throw error
    }
  }

  // Concurrent 401s share one refresh, otherwise rotation would burn the token twice.
  function refreshOnce(): Promise<AuthResponse> {
    inflightRefresh ??= doRefresh().finally(() => {
      inflightRefresh = null
    })
    return inflightRefresh
  }

  async function request<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
    const auth = options.auth ?? true
    let res = await send(path, options, auth ? accessToken : null)

    if (res.status === 401 && auth && storage.getRefresh()) {
      const first = await toApiError(res.clone())
      if (REFRESHABLE.has(first.code)) {
        try {
          await refreshOnce()
        } catch {
          throw first
        }
        res = await send(path, options, accessToken) // exactly one retry
      }
    }

    if (!res.ok) throw await toApiError(res)
    return (res.status === 204 ? undefined : await res.json()) as T
  }

  async function authenticate(path: string, body: unknown): Promise<AuthResponse> {
    return apply(await request<AuthResponse>(path, { method: 'POST', body, auth: false }))
  }

  return {
    request,
    login: (email: string, password: string) => authenticate('/auth/login', { email, password }),
    register: (email: string, password: string, name?: string) =>
      authenticate('/auth/register', { email, password, ...(name ? { name } : {}) }),
    /** Exchange a stored refresh token for a session; null when there is none. */
    async restore(): Promise<AuthResponse | null> {
      if (!storage.getRefresh()) return null
      try {
        return await refreshOnce()
      } catch {
        return null
      }
    },
    /** Forget the session locally, then revoke the refresh token server-side (best effort). */
    async logout(): Promise<void> {
      const token = storage.getRefresh()
      endSession()
      if (!token) return
      try {
        await send('/auth/logout', { method: 'POST', body: { refresh_token: token } }, null)
      } catch {
        /* already signed out locally */
      }
    },
    getAccessToken: () => accessToken,
    onSessionExpired(listener: () => void) {
      expiredListeners.add(listener)
      return () => {
        expiredListeners.delete(listener)
      }
    },
  }
}

export type ApiClient = ReturnType<typeof createApiClient>

export const apiClient: ApiClient = createApiClient({ baseUrl: import.meta.env.VITE_API_BASE ?? '/api/v1' })
