import { useI18n } from '../i18n/context'

type Section = 'library' | 'dictionary'

/** Empty state for a section whose real content arrives in a later milestone. */
export default function SectionPage({ section }: { section: Section }) {
  const { t } = useI18n()
  return (
    <div className="mx-auto max-w-4xl">
      <h2 className="font-display text-2xl font-bold text-white">{t.nav[section]}</h2>
      <p className="mt-2 text-slate-400">{t.pages[section]}</p>
      <div data-testid="page-skeleton" aria-hidden="true" className="mt-8 grid animate-pulse gap-4 sm:grid-cols-2">
        <div className="h-32 rounded-xl bg-white/5" />
        <div className="h-32 rounded-xl bg-white/5" />
        <div className="h-32 rounded-xl bg-white/5 sm:col-span-2" />
      </div>
    </div>
  )
}
