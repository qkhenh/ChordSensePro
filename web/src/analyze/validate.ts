/** Quick checks before anything is sent. The server stays the authority on all of them. */

export const MAX_FILE_BYTES = 100 * 1024 * 1024
export const ACCEPTED_EXTENSIONS = ['mp3', 'wav', 'flac', 'm4a', 'ogg'] as const

export type FileProblem = 'fileEmpty' | 'fileTooLarge' | 'unsupportedFormat'
export type UrlProblem = 'urlInvalid' | 'urlScheme' | 'urlCredentials' | 'urlPrivate' | 'urlPort'

export function validateFile(file: { name: string; size: number }): FileProblem | null {
  if (file.size > MAX_FILE_BYTES) return 'fileTooLarge'
  if (file.size === 0) return 'fileEmpty'
  const dot = file.name.lastIndexOf('.')
  const extension = dot < 0 ? '' : file.name.slice(dot + 1).toLowerCase()
  return (ACCEPTED_EXTENSIONS as readonly string[]).includes(extension) ? null : 'unsupportedFormat'
}

function isPrivateIPv4(host: string): boolean {
  const octets = host.split('.').map(Number)
  const [a, b] = octets
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168)
  )
}

function isInternalHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/\.$/, '')
  if (host === 'localhost' || host.endsWith('.localhost')) return true
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) return isPrivateIPv4(host) // WHATWG URL already normalised 2130706433, 0x7f000001, …
  if (host.startsWith('[')) {
    const v6 = host.slice(1, -1)
    return v6 === '::1' || v6 === '::' || /^f[cd]/.test(v6) || /^fe[89ab]/.test(v6) || v6.startsWith('::ffff:')
  }
  return false
}

export function validateUrl(raw: string): UrlProblem | null {
  const text = raw.trim()
  let url: URL
  try {
    url = new URL(text)
  } catch {
    return 'urlInvalid'
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return 'urlScheme'
  if (url.username || url.password) return 'urlCredentials'
  if (isInternalHost(url.hostname)) return 'urlPrivate'
  if (url.port !== '') return 'urlPort' // the default port is dropped by URL, so any left is non-standard
  return null
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
