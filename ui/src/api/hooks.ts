import { useQuery } from '@tanstack/react-query'
import { request } from './client'
import type { Health } from './types'

export const useHealth = () =>
  useQuery({ queryKey: ['health'], queryFn: () => request<Health>('/health'), retry: 0, refetchInterval: 30_000 })
