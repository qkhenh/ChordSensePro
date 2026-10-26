import { NavLink, Outlet } from 'react-router'
import { useAuth } from '../auth/context'
import { text } from '../auth/messages'
import { useI18n } from '../i18n/context'
import { LANG_NAMES, LANGS } from '../i18n/messages'
import { QuotaBadge } from './QuotaBadge'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-lg px-3 py-2 text-sm font-medium transition ${
    isActive ? 'bg-brand-purple/20 text-white' : 'text-slate-400 hover:bg-white/5 hover:text-white'
  }`

export function ShellLayout() {
  const { t, lang, setLang } = useI18n()
  const { user, logout } = useAuth()

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="border-b border-white/10 p-4 md:w-60 md:shrink-0 md:border-r md:border-b-0">
        <p className="mb-4 font-display text-xl font-extrabold text-white">{text.brand}</p>
        <nav aria-label={t.nav.label} className="flex gap-1 md:flex-col">
          <NavLink to="/library" className={linkClass}>
            {t.nav.library}
          </NavLink>
          <NavLink to="/workspace" className={linkClass}>
            {t.nav.workspace}
          </NavLink>
          <NavLink to="/dictionary" className={linkClass}>
            {t.nav.dictionary}
          </NavLink>
        </nav>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center justify-end gap-3 border-b border-white/10 px-4 py-3">
          <QuotaBadge />
          <div role="group" aria-label={t.header.language} className="flex overflow-hidden rounded-lg border border-white/15">
            {LANGS.map((code) => (
              <button
                key={code}
                type="button"
                aria-pressed={code === lang}
                onClick={() => setLang(code)}
                className={`px-3 py-1 text-sm ${code === lang ? 'bg-white/15 text-white' : 'text-slate-400 hover:bg-white/5'}`}
              >
                {LANG_NAMES[code]}
              </button>
            ))}
          </div>
          <span className="text-sm text-slate-400">{user?.email}</span>
          <button
            type="button"
            onClick={() => void logout()}
            className="rounded-lg border border-white/20 px-3 py-1 text-sm text-white hover:bg-white/10"
          >
            {t.header.logout}
          </button>
        </header>
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
