import { useI18n } from '../i18n/context'
import type { Feature } from './entitlements'

/** Same look as the quota banner: what is locked, and the way to unlock it. */
export function UpgradePrompt({ feature, onDismiss }: { feature: Feature; onDismiss: () => void }) {
  const { t } = useI18n()
  return (
    <div role="status" className="mt-4 rounded-xl border border-amber-400/40 bg-amber-500/10 p-4 text-amber-100">
      <p>{t.practice.upgrade[feature]}</p>
      <div className="mt-2 flex items-center gap-3">
        <a
          href="#upgrade"
          className="rounded-lg bg-amber-400 px-3 py-1.5 text-sm font-semibold text-black hover:bg-amber-300"
        >
          {t.practice.upgrade.cta}
        </a>
        <button type="button" onClick={onDismiss} className="text-sm underline">
          {t.practice.upgrade.dismiss}
        </button>
      </div>
    </div>
  )
}
