import { useAuth } from '../auth/context'
import { text } from '../auth/messages'

/** Placeholder behind the guard; replaced by the Workspace Shell (task 1.5). */
export default function HomePage() {
  const { user, logout } = useAuth()
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <h1 className="font-display text-4xl font-extrabold text-white">{text.brand}</h1>
      <h2 className="text-lg text-slate-300">Workspace</h2>
      <p className="text-slate-400">{user?.email}</p>
      <button
        type="button"
        onClick={() => void logout()}
        className="rounded-lg border border-white/20 px-4 py-2 text-white hover:bg-white/10"
      >
        {text.logout}
      </button>
    </main>
  )
}
