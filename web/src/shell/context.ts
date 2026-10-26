import { createContext, useContext } from 'react'
import type { ToastSpec } from '../api/errors'

export interface ToastApi {
  show(spec: ToastSpec): void
}

export const ToastContext = createContext<ToastApi | null>(null)

export function useToast(): ToastApi {
  const api = useContext(ToastContext)
  if (!api) throw new Error('useToast must be used inside <ToastProvider>')
  return api
}
