import { InlineNotification, Tag } from '@carbon/react'
import { useHealth } from '../api/hooks'

const LABELS: Record<string, string> = {
  forecast: 'Forecast (03_final_forecast.py)',
  accuracy: 'Rolling accuracy (05_rolling_backtest.py)',
  flags: 'Challenge flags (08_challenge_flags.py)',
  explain: 'Driver explanation (10_explain.py)',
}

export default function Overview() {
  const { data, error } = useHealth()
  return (
    <>
      {error ? (
        <InlineNotification
          kind="error"
          lowContrast
          hideCloseButton
          title="The API is not reachable."
          subtitle={error instanceof Error ? error.message : ''}
        />
      ) : null}
      <section className="card">
        <h2 className="card-title">Result files the API can serve</h2>
        <p className="muted">Each file is written by a weekly script. A missing file means that screen has nothing to show yet.</p>
        <ul className="plain-list">
          {Object.entries(data?.results_available ?? {}).map(([key, found]) => (
            <li key={key}>
              <Tag type={found ? 'green' : 'red'}>{found ? 'Found' : 'Missing'}</Tag> {LABELS[key] ?? key}
            </li>
          ))}
        </ul>
      </section>
      <p className="muted">The demand outlook, accuracy by horizon and Why this week cards come in the next steps.</p>
    </>
  )
}
