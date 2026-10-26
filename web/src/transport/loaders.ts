import type { SCNamespace, YTNamespace } from './embeds'

const YOUTUBE_SRC = 'https://www.youtube.com/iframe_api'
const SOUNDCLOUD_SRC = 'https://w.soundcloud.com/player/api.js'
const DEFAULT_TIMEOUT_MS = 10_000

type Win = Window & {
  YT?: YTNamespace
  SC?: SCNamespace
  onYouTubeIframeAPIReady?: () => void
}
const page = () => window as Win

let youtube: Promise<YTNamespace> | null = null
let soundcloud: Promise<SCNamespace> | null = null
let timeoutMs = DEFAULT_TIMEOUT_MS

/** Forget cached loads (tests); a failed load forgets itself so the next call retries. */
export function resetLoaders(options: { timeoutMs?: number } = {}): void {
  youtube = null
  soundcloud = null
  timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
}

function addScript(src: string, onError: () => void): HTMLScriptElement {
  const script = document.createElement('script')
  script.src = src
  script.async = true
  script.addEventListener('error', onError)
  document.head.appendChild(script)
  return script
}

/** The official IFrame API announces itself by calling a global once it is ready. */
export function loadYouTubeApi(): Promise<YTNamespace> {
  const ready = page().YT
  if (ready?.Player) return Promise.resolve(ready)
  youtube ??= new Promise<YTNamespace>((resolve, reject) => {
    const fail = (reason: string) => {
      clearTimeout(timer)
      script.remove()
      youtube = null
      reject(new Error(reason))
    }
    const timer = setTimeout(() => fail('Loading the YouTube player timed out'), timeoutMs)
    const previous = page().onYouTubeIframeAPIReady
    page().onYouTubeIframeAPIReady = () => {
      previous?.()
      clearTimeout(timer)
      const api = page().YT
      if (api) resolve(api)
      else fail('The YouTube player loaded but did not provide its API')
    }
    const script = addScript(YOUTUBE_SRC, () => fail('The YouTube player script could not be loaded'))
  })
  return youtube
}

/** The Widget API is a plain script: it is ready when the script has loaded. */
export function loadSoundCloudApi(): Promise<SCNamespace> {
  const ready = page().SC
  if (ready?.Widget) return Promise.resolve(ready)
  soundcloud ??= new Promise<SCNamespace>((resolve, reject) => {
    const fail = (reason: string) => {
      clearTimeout(timer)
      script.remove()
      soundcloud = null
      reject(new Error(reason))
    }
    const timer = setTimeout(() => fail('Loading the SoundCloud player timed out'), timeoutMs)
    const script = addScript(SOUNDCLOUD_SRC, () => fail('The SoundCloud player script could not be loaded'))
    script.addEventListener('load', () => {
      clearTimeout(timer)
      const api = page().SC
      if (api) resolve(api)
      else fail('The SoundCloud player loaded but did not provide its API')
    })
  })
  return soundcloud
}
