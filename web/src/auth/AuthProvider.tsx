import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import type { ApiClient } from '../api/http'
import { AuthContext, type AuthContextValue, type AuthStatus, type AuthUser } from './context'

export function AuthProvider({ client, children }: { client: ApiClient; children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [user, setUser] = useState<AuthUser | null>(null)
  const queries = useQueryClient() // what was fetched for one person must not outlive their session

  useEffect(() => {
    let active = true
    const signedOut = () => {
      queries.clear()
      setUser(null)
      setStatus('anonymous')
    }
    client.restore().then((session) => {
      if (!active) return
      if (session) {
        setUser(session.user)
        setStatus('authenticated')
      } else {
        signedOut()
      }
    })
    const unsubscribe = client.onSessionExpired(signedOut)
    return () => {
      active = false
      unsubscribe()
    }
  }, [client, queries])

  const value = useMemo<AuthContextValue>(() => {
    const signIn = (session: { user: AuthUser }) => {
      setUser(session.user)
      setStatus('authenticated')
    }
    return {
      status,
      user,
      login: async (email, password) => signIn(await client.login(email, password)),
      register: async (email, password, name) => signIn(await client.register(email, password, name)),
      logout: async () => {
        await client.logout()
        queries.clear()
        setUser(null)
        setStatus('anonymous')
      },
    }
  }, [client, queries, status, user])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
