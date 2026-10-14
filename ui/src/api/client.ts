import { env } from '../config/env'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

/** One place where every API call is made, so errors read the same everywhere. */
export async function request<T>(path: string): Promise<T> {
  let res: Response
  try {
    res = await fetch(env.apiUrl + path)
  } catch {
    throw new ApiError(`The API is not reachable. Start it from the api folder with: ${env.apiStartCommand}`, 0)
  }
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      /* the body was not JSON */
    }
    throw new ApiError(detail || `API ${path} returned ${res.status}`, res.status)
  }
  return (await res.json()) as T
}
