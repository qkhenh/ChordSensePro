import { ApiError } from '../api/http'
import { errorMessage } from '../auth/messages'
import type { Messages } from '../i18n/messages'

/** Why a request to start an analysis was refused, in the user's language. */
export function analyzeErrorText(t: Messages, error: unknown): string {
  const known = t.analyze.errors as Record<string, string>
  if (error instanceof ApiError && error.code in known && error.code !== 'unknown') {
    return known[error.code]
  }
  return errorMessage(error)
}
