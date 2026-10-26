import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { ToastSpec } from '../api/errors'
import { useI18n } from '../i18n/context'
import { ToastContext } from './context'

const LIFETIME_MS = 5000

interface ToastItem extends ToastSpec {
  id: number
}

const STYLE: Record<ToastSpec['kind'], string> = {
  error: 'border-red-400/40 bg-red-950/90 text-red-100',
  warning: 'border-amber-400/40 bg-amber-950/90 text-amber-100',
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const { t } = useI18n()
  const [items, setItems] = useState<ToastItem[]>([])
  const nextId = useRef(0)
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>())

  const dismiss = useCallback((id: number) => {
    clearTimeout(timers.current.get(id))
    timers.current.delete(id)
    setItems((current) => current.filter((item) => item.id !== id))
  }, [])

  const show = useCallback(
    (spec: ToastSpec) => {
      const id = nextId.current++
      setItems((current) => [...current, { ...spec, id }])
      timers.current.set(id, setTimeout(() => dismiss(id), LIFETIME_MS))
    },
    [dismiss],
  )

  useEffect(() => {
    const pending = timers.current
    return () => pending.forEach(clearTimeout)
  }, [])

  const api = useMemo(() => ({ show }), [show])

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="pointer-events-none fixed inset-x-4 bottom-4 z-50 flex flex-col items-end gap-2">
        {items.map((item) => (
          <div
            key={item.id}
            role={item.kind === 'error' ? 'alert' : 'status'}
            className={`pointer-events-auto flex max-w-sm items-start gap-3 rounded-lg border px-4 py-3 text-sm shadow-xl ${STYLE[item.kind]}`}
          >
            <span className="flex-1">{item.message}</span>
            <button
              type="button"
              aria-label={t.toast.close}
              onClick={() => dismiss(item.id)}
              className="-mr-1 rounded px-1 opacity-70 hover:opacity-100"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
