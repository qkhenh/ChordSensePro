import { useI18n } from '../i18n/context'

interface Props {
  /** 0-based bar indexes of the loop, both ends included. */
  range: { from: number; to: number } | null
  onClear: () => void
}

export function LoopControls({ range, onClear }: Props) {
  const { t } = useI18n()
  if (!range) return <p className="text-sm text-slate-500">{t.practice.loopHint}</p>
  return (
    <div className="flex items-center gap-3 text-sm text-slate-300">
      <span>{t.practice.loopLabel(range.from + 1, range.to + 1)}</span>
      <button
        type="button"
        onClick={onClear}
        className="rounded-md border border-white/10 px-2 py-1 hover:bg-white/10"
      >
        {t.practice.clearLoop}
      </button>
    </div>
  )
}
