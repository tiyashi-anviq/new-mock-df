import type { ReactNode } from 'react'

interface Props {
  group: string
  title: string
  subtitle: string
  right?: ReactNode
}

export default function PageHeader({ group, title, subtitle, right }: Props) {
  return (
    <header className="page-head">
      <div>
        <p className="page-crumb">{group}</p>
        <h1 className="page-title">{title}</h1>
        <p className="page-sub">{subtitle}</p>
      </div>
      {right ? <div className="page-head-right">{right}</div> : null}
    </header>
  )
}
