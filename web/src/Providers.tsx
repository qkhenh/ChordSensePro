import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { Outlet } from 'react-router'
import { ApiClientContext } from './api/context'
import type { ApiClient } from './api/http'
import { AuthProvider } from './auth/AuthProvider'
import { I18nProvider } from './i18n/I18nProvider'
import { ToastProvider } from './shell/toast'

export function Providers({ client }: { client: ApiClient }) {
  const [queries] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))
  return (
    <I18nProvider>
      <ToastProvider>
        <ApiClientContext.Provider value={client}>
          <QueryClientProvider client={queries}>
            <AuthProvider client={client}>
              <Outlet />
            </AuthProvider>
          </QueryClientProvider>
        </ApiClientContext.Provider>
      </ToastProvider>
    </I18nProvider>
  )
}
