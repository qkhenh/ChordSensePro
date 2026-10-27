import { useInfiniteQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { Link } from 'react-router'
import { listCharts } from '../api/analyze'
import { useApiClient } from '../api/context'
import { toastFor } from '../api/errors'
import type { ChartSummary } from '../api/types'
import { minutesAndSeconds } from '../format/duration'
import { useI18n } from '../i18n/context'
import { useToast } from '../shell/context'


type SourceLabel = 'file' | 'url' | 'demo' | 'recording'
const sourceLabel = (kind: string): SourceLabel =>
  kind === 'url' || kind === 'demo' || kind === 'recording' ? kind : 'file'

function ChartCard({ chart }: { chart: ChartSummary }) {
  const { t } = useI18n()
  return (
    <li>
      <Link
        to={`/charts/${chart.id}`}
        className="block rounded-xl border border-white/10 bg-white/5 px-4 py-3 transition hover:bg-white/10"
      >
        <span className="flex items-baseline justify-between gap-3">
          <span className="font-semibold text-white">{chart.title}</span>
          <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">
            {t.library.source[sourceLabel(chart.source_kind)]}
          </span>
        </span>
        {chart.artist && <span className="block text-sm text-slate-400">{chart.artist}</span>}
        <span className="mt-1 flex flex-wrap gap-x-3 text-sm text-slate-400">
          <span>{minutesAndSeconds(chart.duration_ms)}</span>
          <span>{Math.round(chart.tempo_bpm)} BPM</span>
          <span>{chart.key}</span>
          {chart.is_customized && <span>{t.library.customized}</span>}
        </span>
      </Link>
    </li>
  )
}

export default function LibraryPage() {
  const client = useApiClient()
  const { t } = useI18n()
  const { show } = useToast()
  const query = useInfiniteQuery({
    queryKey: ['charts'],
    queryFn: ({ pageParam }) => listCharts(client, { cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    retry: false,
  })

  useEffect(() => {
    if (query.error) show(toastFor(query.error))
  }, [query.error, show])

  const charts = query.data?.pages.flatMap((page) => page.items) ?? []
  return (
    <div className="mx-auto max-w-4xl">
      <h2 className="font-display text-2xl font-bold text-white">{t.nav.library}</h2>
      {query.isPending && !query.isError ? (
        <div
          data-testid="library-skeleton"
          role="status"
          aria-label={t.library.loading}
          className="mt-6 grid animate-pulse gap-3 sm:grid-cols-2"
        >
          <div className="h-20 rounded-xl bg-white/5" />
          <div className="h-20 rounded-xl bg-white/5" />
          <div className="h-20 rounded-xl bg-white/5" />
        </div>
      ) : query.isError ? null : charts.length === 0 ? (
        <div className="mt-6 text-slate-400">
          <p>{t.library.empty}</p>
          <Link to="/workspace" className="mt-2 inline-block text-brand-purple-light underline">
            {t.library.start}
          </Link>
        </div>
      ) : (
        <>
          <ul className="mt-6 grid gap-3 sm:grid-cols-2">
            {charts.map((chart) => (
              <ChartCard key={chart.id} chart={chart} />
            ))}
          </ul>
          {query.hasNextPage && (
            <button
              type="button"
              disabled={query.isFetchingNextPage}
              onClick={() => void query.fetchNextPage()}
              className="mt-4 rounded-lg border border-white/10 px-4 py-2 text-sm text-slate-200 hover:bg-white/10 disabled:opacity-50"
            >
              {t.library.loadMore}
            </button>
          )}
        </>
      )}
    </div>
  )
}
