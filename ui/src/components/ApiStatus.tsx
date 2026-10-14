import { SkeletonText, Tag } from '@carbon/react'
import { useHealth } from '../api/hooks'

/** A one-line proof that the UI can reach the API. */
export default function ApiStatus() {
  const { data, error, isPending } = useHealth()
  if (isPending) return <SkeletonText width="14rem" />
  if (error || !data) {
    return (
      <Tag type="red" title={error instanceof Error ? error.message : undefined}>
        API not reachable
      </Tag>
    )
  }
  return (
    <Tag type="green">
      API connected · {data.series} series · history to {data.last_history_week}
    </Tag>
  )
}
