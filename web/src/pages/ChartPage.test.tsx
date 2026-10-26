import { render, screen } from '@testing-library/react'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { describe, expect, it } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { Chart } from '../api/types'
import { buildRoutes } from '../routes'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'

const API = '*/api/v1'
const chart = Object.values(
  import.meta.glob<Chart>('../../../contracts/fixtures/charts/ii_v_i_c.chart.json', {
    eager: true,
    import: 'default',
  }),
)[0]

function renderChart(id: string) {
  server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage('refresh-stored') })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: [`/charts/${id}`] })
  render(<RouterProvider router={router} />)
  return router
}

describe('chart page', () => {
  it('shows the song, its key and tempo and every chord with its start time', async () => {
    server.use(http.get(`${API}/charts/${chart.id}`, () => HttpResponse.json(chart)))
    renderChart(chart.id)

    expect(await screen.findByRole('heading', { level: 2, name: chart.title })).toBeInTheDocument()
    expect(screen.getByText(/C major/)).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`${Math.round(chart.metadata.beat_grid.tempo_bpm)}`))).toBeInTheDocument()

    const rows = screen.getAllByRole('row').slice(1) // without the header
    expect(rows).toHaveLength(chart.ai_events.length)
    expect(rows[0]).toHaveTextContent(chart.ai_events[0].symbol)
    expect(rows[0]).toHaveTextContent('0:00')
  })

  it('formats start times as minutes and seconds', async () => {
    const long = {
      ...chart,
      ai_events: chart.ai_events.map((event, i) => ({
        ...event,
        start_ms: i === 1 ? 65_000 : event.start_ms,
      })),
    }
    server.use(http.get(`${API}/charts/${chart.id}`, () => HttpResponse.json(long)))
    renderChart(chart.id)
    expect(await screen.findByText('1:05')).toBeInTheDocument()
  })

  it('says when the chart does not exist and leads back to the library', async () => {
    server.use(http.get(`${API}/charts/nope`, () => apiError(404, 'not_found')))
    renderChart('nope')
    expect(await screen.findByText('Không tìm thấy bài này')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Về Thư viện' })).toHaveAttribute('href', '/library')
  })

  it('shows a loading state, then reports a server problem without crashing', async () => {
    server.use(http.get(`${API}/charts/boom`, () => apiError(500, 'internal_error')))
    renderChart('boom')
    expect(await screen.findByRole('alert')).toHaveTextContent(/không tải được/i)
  })
})
