import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { analyzeDemo, listDemoTracks } from '../api/analyze'
import { useApiClient } from '../api/context'
import { toastFor } from '../api/errors'
import { useI18n } from '../i18n/context'
import { useToast } from '../shell/context'

/** UC1 V1: try the product on a ready-made song. Free, so it never looks at the quota. */
export function DemoTracks({ onStart }: { onStart(jobId: string): void }) {
  const client = useApiClient()
  const { t } = useI18n()
  const { show } = useToast()

  const catalog = useQuery({
    queryKey: ['demo-tracks'],
    queryFn: () => listDemoTracks(client),
    staleTime: Infinity,
    retry: false,
  })
  const start = useMutation({
    mutationFn: (id: string) => analyzeDemo(client, id),
    onSuccess: ({ job_id }) => onStart(job_id),
    onError: (error) => show(toastFor(error)),
  })

  useEffect(() => {
    if (catalog.error) show(toastFor(catalog.error))
  }, [catalog.error, show])

  if (catalog.isError) return null
  return (
    <section className="mt-8">
      <h3 className="font-display text-lg font-semibold text-white">{t.demo.heading}</h3>
      <p className="mb-3 text-sm text-slate-500">{t.demo.hint}</p>
      {catalog.isPending ? (
        <div data-testid="demo-skeleton" aria-hidden="true" className="grid animate-pulse gap-3 sm:grid-cols-2">
          <div className="h-14 rounded-xl bg-white/5" />
          <div className="h-14 rounded-xl bg-white/5" />
          <div className="h-14 rounded-xl bg-white/5" />
        </div>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2">
          {catalog.data.map((track) => (
            <li key={track.id}>
              <button
                type="button"
                disabled={start.isPending}
                onClick={() => start.mutate(track.id)}
                className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-left transition hover:bg-white/10 disabled:opacity-50"
              >
                <span className="block font-medium text-white">{track.title}</span>
                <span className="block text-sm text-slate-400">
                  {track.key} · {Math.round(track.tempo_bpm)} {t.chart.bpm}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
