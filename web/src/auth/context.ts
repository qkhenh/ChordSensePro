import { createContext, useContext } from 'react'
import type { AuthResponse } from '../api/http'

export type AuthStatus = 'loading' | 'authenticated' | 'anonymous'
export type AuthUser = AuthResponse['user']

export interface AuthContextValue {
  status: AuthStatus
  user: AuthUser | null
  login(email: string, password: string): Promise<void>
  register(email: string, password: string, name?: string): Promise<void>
  logout(): Promise<void>
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside <AuthProvider>')
  return value
}
