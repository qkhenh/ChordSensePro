import { useQuery } from '@tanstack/react-query'
import { useApiClient } from '../api/context'
import type { Quota } from '../api/types'
import { useAuth } from '../auth/context'

export const quotaKey = (userId: string | undefined) => ['quota', userId] as const

export function useQuota() {
  const client = useApiClient()
  const { user } = useAuth()
  return useQuery({
    queryKey: quotaKey(user?.id),
    queryFn: () => client.request<Quota>('/users/me/quota'),
    enabled: !!user,
    staleTime: 30_000,
  })
}
