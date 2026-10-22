import type { ReactNode } from 'react'
import { text } from '../auth/messages'

export function AuthLayout({ title, children }: { title: string; children: ReactNode }) {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-6 py-12">
      <header className="text-center">
        <h1 className="font-display text-4xl font-extrabold tracking-tight text-white">{text.brand}</h1>
        <p className="mt-2 max-w-md text-slate-400">{text.tagline}</p>
      </header>
      <section className="w-full max-w-sm rounded-2xl border border-white/10 bg-white/5 p-6 shadow-xl">
        <h2 className="mb-4 text-xl font-semibold text-white">{title}</h2>
        {children}
      </section>
    </main>
  )
}

export function Field({
  id,
  label,
  error,
  ...input
}: { id: string; label: string; error?: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="mb-4">
      <label htmlFor={id} className="mb-1 block text-sm font-medium text-slate-300">
        {label}
      </label>
      <input
        id={id}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${id}-error` : undefined}
        className="w-full rounded-lg border border-white/15 bg-black/30 px-3 py-2 text-white outline-none focus:border-brand-purple-light"
        {...input}
      />
      {error && (
        <p id={`${id}-error`} className="mt-1 text-sm text-red-400">
          {error}
        </p>
      )}
    </div>
  )
}

export function SubmitButton({ pending, idle, busy }: { pending: boolean; idle: string; busy: string }) {
  return (
    <button
      type="submit"
      disabled={pending}
      className="w-full rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white transition hover:bg-brand-purple-light disabled:opacity-60"
    >
      {pending ? busy : idle}
    </button>
  )
}

export function FormError({ message }: { message: string | null }) {
  if (!message) return null
  return (
    <p role="alert" className="mb-4 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
      {message}
    </p>
  )
}
