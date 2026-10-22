/** Where a visitor was heading before RequireAuth sent them to /login. */
export function destinationFrom(state: unknown): string {
  return (state as { from?: { pathname?: string } } | null)?.from?.pathname ?? '/'
}
