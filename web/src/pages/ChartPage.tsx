import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router'
import { getChart } from '../api/analyze'
import { useApiClient } from '../api/context'
import { ApiError } from '../api/http'
import { useI18n } from '../i18n/context'

const minutesAndSeconds = (ms: number) => {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

/** Plain read-out of a chart. The interactive Smart Grid replaces this in Milestone 3. */
export default function ChartPage() {
  const { id = '' } = useParams()
  const client = useApiClient()
  const { t } = useI18n()
  const { data: chart, error, isPending } = useQuery({
    queryKey: ['chart', id],
    queryFn: () => getChart(client, id),
    retry: false,
  })

  if (isPending) {
    return (
      <p role="status" className="text-slate-400">
        {t.chart.loading}
      </p>
    )
  }
  if (error instanceof ApiError && error.status === 404) {
    return (
      <div className="text-slate-300">
        <p>{t.chart.notFound}</p>
        <Link to="/library" className="mt-2 inline-block text-brand-purple-light underline">
          {t.chart.back}
        </Link>
      </div>
    )
  }
  if (error || !chart) {
    return (
      <p role="alert" className="rounded-lg bg-red-500/10 px-3 py-2 text-red-300">
        {t.chart.unavailable}
      </p>
    )
  }

  const { key, beat_grid: grid } = chart.metadata
  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="font-display text-2xl font-bold text-white">{chart.title}</h2>
      <p className="mt-2 text-slate-400">
        {t.chart.key}: {key.tonic} {key.mode} · {t.chart.tempo}: {Math.round(grid.tempo_bpm)} {t.chart.bpm}
      </p>
      <table className="mt-6 w-full text-left text-slate-200">
        <thead className="text-sm text-slate-500">
          <tr>
            <th className="py-2 font-medium">{t.chart.time}</th>
            <th className="py-2 font-medium">{t.chart.chord}</th>
          </tr>
        </thead>
        <tbody>
          {chart.ai_events.map((event) => (
            <tr key={event.event_id} className="border-t border-white/10">
              <td className="py-2 text-slate-400">{minutesAndSeconds(event.start_ms)}</td>
              <td className="py-2 font-semibold">{event.symbol}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
