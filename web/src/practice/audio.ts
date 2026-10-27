let shared: AudioContext | null = null

/**
 * The page's one AudioContext, made on first use. Browsers only let it run after a user gesture,
 * so it is created from a click handler and resumed there (see UnlockModal).
 */
export function ensureAudioContext(): AudioContext | null {
  if (shared) return shared
  const Ctor =
    globalThis.AudioContext ?? (globalThis as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!Ctor) return null
  try {
    shared = new Ctor()
  } catch {
    return null
  }
  return shared
}

/** Tests only: forget the context so the next test builds its own. */
export function resetAudioContext(): void {
  shared = null
}
