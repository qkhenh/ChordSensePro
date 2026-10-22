import { HttpResponse, http } from 'msw'
import { setupServer } from 'msw/node'
import type { components } from '../api/generated/schema'

type AuthResponse = components['schemas']['AuthResponse']

export const server = setupServer()

export const TEST_USER: AuthResponse['user'] = {
  id: '11111111-1111-4111-8111-111111111111',
  email: 'ann@example.com',
  name: null,
  role: 'free',
  created_at: '2026-10-22T09:00:00Z',
}

export function authResponse(overrides: Partial<AuthResponse> = {}): AuthResponse {
  return {
    user: TEST_USER,
    access_token: 'access-1',
    refresh_token: 'refresh-1',
    token_type: 'bearer',
    expires_in: 900,
    ...overrides,
  }
}

export function apiError(status: number, code: string, message = code) {
  return HttpResponse.json({ code, message }, { status })
}

export { HttpResponse, http }
