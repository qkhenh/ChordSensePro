import { getAudioTicket } from '../api/analyze'
import type { ApiClient } from '../api/http'
import type { PlaybackSource } from '../api/types'
import type { SCNamespace, YTNamespace } from './embeds'
import { HtmlAudioTransport } from './html'
import { loadSoundCloudApi, loadYouTubeApi } from './loaders'
import { SoundCloudTransport } from './soundcloud'
import type { Transport } from './types'
import { YouTubeTransport } from './youtube'

export interface TransportDeps {
  chartId: string
  client: ApiClient
  /** Where players that need a place on the page (YouTube, SoundCloud) are put. */
  mount: HTMLElement
  /** Replaceable in tests so no real script is fetched. */
  loaders?: {
    youtube?: () => Promise<YTNamespace>
    soundcloud?: () => Promise<SCNamespace>
  }
}

/** The transport that can play this chart's source. Elements it adds to `mount` go with `destroy()`. */
export function createTransport(playback: PlaybackSource, deps: TransportDeps): Transport {
  switch (playback.kind) {
    case 'file':
      return new HtmlAudioTransport(
        new Audio(),
        async () => (await getAudioTicket(deps.client, deps.chartId)).url,
      )
    case 'direct':
      return new HtmlAudioTransport(new Audio(), async () => playback.audio_url ?? '')
    case 'youtube': {
      const host = document.createElement('div')
      deps.mount.appendChild(host)
      const transport = new YouTubeTransport(
        host,
        playback.embed_id ?? '',
        deps.loaders?.youtube ?? loadYouTubeApi,
      )
      return withCleanup(transport, () => host.remove())
    }
    case 'soundcloud': {
      const frame = document.createElement('iframe')
      frame.title = 'SoundCloud'
      frame.allow = 'autoplay'
      frame.src = `https://w.soundcloud.com/player/?url=${encodeURIComponent(playback.embed_id ?? '')}`
      deps.mount.appendChild(frame)
      const transport = new SoundCloudTransport(frame, deps.loaders?.soundcloud ?? loadSoundCloudApi)
      return withCleanup(transport, () => frame.remove())
    }
  }
}

/** Run `after` once the transport is destroyed (to take its elements off the page). */
function withCleanup<T extends Transport>(transport: T, after: () => void): T {
  const destroy = transport.destroy.bind(transport)
  transport.destroy = () => {
    destroy()
    after()
  }
  return transport
}
