import { useEffect, useRef } from 'react'
import { useI18n } from '../i18n/context'

/** UC2 2a: the browser refused to start sound by itself, so ask for one explicit click. */
export function UnlockModal({ onUnlock }: { onUnlock: () => void }) {
  const { t } = useI18n()
  const button = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    button.current?.focus()
  }, [])
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="unlock-title"
        className="w-full max-w-sm rounded-2xl border border-white/10 bg-[#0b0b1a] p-6 text-center"
      >
        <h3 id="unlock-title" className="font-display text-lg font-semibold text-white">
          {t.practice.unlockTitle}
        </h3>
        <p className="mt-2 text-sm text-slate-400">{t.practice.unlockBody}</p>
        <button
          ref={button}
          type="button"
          onClick={onUnlock}
          className="mt-5 rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white hover:opacity-90"
        >
          {t.practice.unlock}
        </button>
      </div>
    </div>
  )
}
