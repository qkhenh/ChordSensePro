// The job being watched survives a page reload (F5) as long as the tab lives.
const KEY = 'chordsense.activeJob'

// sessionStorage can be missing or throw (private mode, blocked site data).
export function readActiveJob(): string | null {
  try {
    return sessionStorage.getItem(KEY)
  } catch {
    return null
  }
}

export function writeActiveJob(jobId: string): void {
  try {
    sessionStorage.setItem(KEY, jobId)
  } catch {
    /* tracking simply will not survive a reload */
  }
}

export function clearActiveJob(): void {
  try {
    sessionStorage.removeItem(KEY)
  } catch {
    /* nothing to forget */
  }
}
