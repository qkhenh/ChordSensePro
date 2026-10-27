import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import { authResponse, http, HttpResponse, server } from '../test/server'
import { AuthProvider } from './AuthProvider'
import { useAuth } from './context'

function Buttons() {
  const { login, logout, status } = useAuth()
  return (
    <>
      <output>{status}</output>
      <button onClick={() => void login('ann@example.com', 'secret-pass')}>in</button>
      <button onClick={() => void logout()}>out</button>
    </>
  )
}

describe('signing out', () => {
  it('forgets everything fetched for that person, so the next one on this machine never sees it', async () => {
    server.use(
      http.post('*/api/v1/auth/login', () => HttpResponse.json(authResponse())),
      http.post('*/api/v1/auth/logout', () => new HttpResponse(null, { status: 204 })),
    )
    const client = createApiClient({ storage: memoryStorage() })
    const queries = new QueryClient()
    render(
      <QueryClientProvider client={queries}>
        <AuthProvider client={client}>
          <Buttons />
        </AuthProvider>
      </QueryClientProvider>,
    )
    await userEvent.click(await screen.findByRole('button', { name: 'in' }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('authenticated'))
    queries.setQueryData(['charts'], { pages: [{ items: [{ title: 'A secret song' }] }] })
    queries.setQueryData(['chart', 'x'], { title: 'A secret song' })
    await userEvent.click(screen.getByRole('button', { name: 'out' }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('anonymous'))
    expect(queries.getQueryCache().getAll()).toHaveLength(0)
  })
})
