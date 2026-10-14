// The shapes the API returns. They mirror the response models in api/app.py.
export interface Health {
  status: 'ok'
  series: number
  national_series: number
  kolkata_series: number
  first_week: string
  last_history_week: string
  weeks: number
  results_available: Record<string, boolean>
}
