import { describe, expect, it } from 'vitest'
import { formatBytes, MAX_FILE_BYTES, validateFile, validateUrl } from './validate'

const file = (name: string, size = 1000) => ({ name, size })

describe('validateFile', () => {
  it.each(['song.mp3', 'SONG.WAV', 'a.b.flac', 'x.m4a', 'x.ogg'])('accepts %s', (name) => {
    expect(validateFile(file(name))).toBeNull()
  })

  it('accepts a file of exactly the limit and refuses one byte more', () => {
    expect(validateFile(file('a.mp3', MAX_FILE_BYTES))).toBeNull()
    expect(validateFile(file('a.mp3', MAX_FILE_BYTES + 1))).toBe('fileTooLarge')
  })

  it('refuses an empty file', () => {
    expect(validateFile(file('a.mp3', 0))).toBe('fileEmpty')
  })

  it.each(['notes.txt', 'movie.mp4', 'archive.zip', 'noextension', '.mp3x', 'song.mp3.exe'])(
    'refuses %s',
    (name) => {
      expect(validateFile(file(name))).toBe('unsupportedFormat')
    },
  )

  it('reports the size problem before the format problem', () => {
    expect(validateFile(file('notes.txt', MAX_FILE_BYTES + 1))).toBe('fileTooLarge')
  })
})

describe('validateUrl', () => {
  it.each([
    'https://youtu.be/dQw4w9WgXcQ',
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'https://soundcloud.com/artist/track',
    'http://cdn.example.com/a/song.mp3',
    'https://93.184.216.34/song.mp3',
    '  https://example.com/a.mp3  ',
  ])('accepts %s', (url) => {
    expect(validateUrl(url)).toBeNull()
  })

  it.each(['', '   ', 'not a url', 'example.com/song.mp3', 'http://'])('%j is not a link', (url) => {
    expect(validateUrl(url)).toBe('urlInvalid')
  })

  it.each(['ftp://example.com/a.mp3', 'file:///etc/passwd', 'javascript:alert(1)', 'data:audio/mp3;base64,AA'])(
    '%s has the wrong scheme',
    (url) => {
      expect(validateUrl(url)).toBe('urlScheme')
    },
  )

  it('refuses embedded credentials', () => {
    expect(validateUrl('https://user:pass@example.com/a.mp3')).toBe('urlCredentials')
    expect(validateUrl('https://user@example.com/a.mp3')).toBe('urlCredentials')
  })

  it.each([
    'http://localhost/a.mp3',
    'http://LOCALHOST:80/a.mp3',
    'http://api.localhost/a.mp3',
    'http://127.0.0.1/a.mp3',
    'http://2130706433/a.mp3',
    'http://0x7f000001/a.mp3',
    'http://10.0.0.5/a.mp3',
    'http://172.16.0.1/a.mp3',
    'http://192.168.1.1/a.mp3',
    'http://169.254.169.254/latest/meta-data/',
    'http://0.0.0.0/a.mp3',
    'http://[::1]/a.mp3',
    'http://[fd00::1]/a.mp3',
    'http://[fe80::1]/a.mp3',
  ])('%s points at an internal address', (url) => {
    expect(validateUrl(url)).toBe('urlPrivate')
  })

  it('only accepts the standard ports', () => {
    expect(validateUrl('https://example.com:443/a.mp3')).toBeNull()
    expect(validateUrl('https://example.com:8443/a.mp3')).toBe('urlPort')
    expect(validateUrl('http://example.com:22/a.mp3')).toBe('urlPort')
  })
})

describe('formatBytes', () => {
  it.each([
    [4, '4 B'],
    [1536, '1.5 KB'],
    [5 * 1024 * 1024, '5.0 MB'],
  ])('%d bytes read as %s', (bytes, text) => {
    expect(formatBytes(bytes)).toBe(text)
  })
})
