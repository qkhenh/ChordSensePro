import { vi } from 'vitest'
import { FakeAudio } from '../test/fakeMedia'
import { runTransportContract } from '../test/transportContract'
import { HtmlAudioTransport } from './html'

runTransportContract(
  'html audio',
  () => {
    const audio = new FakeAudio()
    const source = vi.fn().mockResolvedValue('/api/v1/charts/1/audio?ticket=t1')
    const transport = new HtmlAudioTransport(audio as unknown as HTMLAudioElement, source)
    return {
      transport,
      driver: {
        finishLoading: (ms) => audio.finishLoading(ms),
        breakDown: () => {
          source.mockRejectedValue(new Error('offline')) // the retry cannot get a new ticket either
          audio.fail(2)
        },
        rate: () => audio.playbackRate,
      },
    }
  },
  { rate: true },
)
