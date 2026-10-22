import { useEffect, useMemo, useState, type ReactNode } from 'react'
import type { ApiClient } from '../api/http'
import { AuthContext, type AuthContextValue, type AuthStatus, type AuthUser } from './context'

export function AuthProvider({ client, children }: { client: ApiClient; children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [user, setUser] = useState<AuthUser | null>(null)

  useEffect(() => {
    let active = true
    const signedOut = () => {
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
  }, [client])

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
        setUser(null)
        setStatus('anonymous')
      },
    }
  }, [client, status, user])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
