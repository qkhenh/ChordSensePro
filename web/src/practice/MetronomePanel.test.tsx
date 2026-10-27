import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import { I18nProvider } from '../i18n/I18nProvider'
import { scheduleClick } from '../metronome/click'
import type { PlaybackClock } from '../metronome/scheduler'
import type { BeatSource } from '../metronome/beats'
import type { Transport } from '../transport/types'
import { resetAudioContext } from './audio'
import { PracticeController } from './controller'
import { MetronomePanel } from './MetronomePanel'
import { useMetronome } from './useMetronome'

interface FakeScheduler {
  ctx: unknown
  source: BeatSource
  clock: PlaybackClock
  opts: { timbre?: string; volume?: number }
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
  reset: ReturnType<typeof vi.fn>
  setTimbre: ReturnType<typeof vi.fn>
  setVolume: ReturnType<typeof vi.fn>
}
const made = vi.hoisted(() => ({ schedulers: [] as unknown[] }))
vi.mock('../metronome/scheduler', () => ({
  MetronomeScheduler: class {
    start = vi.fn()
    stop = vi.fn()
    reset = vi.fn()
    setTimbre = vi.fn()
    setVolume = vi.fn()
    constructor(ctx: unknown, source: unknown, clock: unknown, opts: unknown) {
      Object.assign(this, { ctx, source, clock, opts })
      made.schedulers.push(this)
    }
  },
}))
vi.mock('../metronome/click', () => ({ scheduleClick: vi.fn(() => () => {}) }))

const schedulers = () => made.schedulers as FakeScheduler[]
const latest = () => schedulers()[schedulers().length - 1]

const meta: AnalysisMetadata = {
  duration_ms: 40_000,
  key: { tonic: 'C', mode: 'major', confidence: 0.9 },
  beat_grid: {
    tempo_bpm: 120,
    tempo_confidence: 0.9,
    time_signature: { numerator: 4, denominator: 4 },
    beats_per_bar: 4,
    first_beat_offset_ms: 500,
    downbeats_ms: null,
  },
  model_version: 'm',
  pipeline_version: 'p',
}
const grid = buildGrid(meta, [])

const ctx = { currentTime: 12.5, state: 'suspended', resume: vi.fn(async () => {}), close: vi.fn(async () => {}) }
const transport = {
  state: 'paused',
  pos: 4321,
  currentTimeMs() {
    return this.pos
  },
  play: vi.fn(async () => {}),
  capabilities: { rate: true },
}
const seeks: number[] = []
const controller = new PracticeController(
  { seek: (ms) => seeks.push(ms), currentTimeMs: () => transport.pos },
  grid,
)

function Harness({ rate = 1 }: { rate?: 0.5 | 0.75 | 1 }) {
  const metronome = useMetronome({ transport: transport as unknown as Transport, grid, controller, rate, tempoBpm: 120, beatsPerBar: 4 })
  return (
    <I18nProvider>
      <MetronomePanel metronome={metronome} />
      <button
        type="button"
        onClick={() => {
          void metronome.runCountIn().then((go) => {
            if (go) void transport.play()
          })
        }}
      >
        go
      </button>
    </I18nProvider>
  )
}

beforeEach(() => {
  made.schedulers.length = 0
  transport.state = 'paused'
  transport.pos = 4321
  transport.play.mockClear()
  ctx.currentTime = 12.5
  ctx.resume.mockClear()
  vi.mocked(scheduleClick).mockClear()
  vi.stubGlobal('AudioContext', class { constructor() { return ctx } })
})
afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
  resetAudioContext()
})

const enable = () => userEvent.click(screen.getByRole('switch', { name: 'Metronome' }))

describe('switching on', () => {
  it('starts a scheduler on the shared audio context after resuming it', async () => {
    render(<Harness />)
    expect(schedulers()).toHaveLength(0)
    await enable()
    expect(ctx.resume).toHaveBeenCalled()
    expect(schedulers()).toHaveLength(1)
    expect(latest().ctx).toBe(ctx)
    expect(latest().start).toHaveBeenCalled()
    expect(screen.getByRole('switch', { name: 'Metronome' })).toBeChecked()
  })

  it('manual mode runs on its own clock, whatever the song is doing', async () => {
    render(<Harness />)
    await enable()
    const first = latest().clock.sample()
    expect(first).toMatchObject({ songMs: 0, rate: 1, playing: true })
    ctx.currentTime += 2
    expect(latest().clock.sample().songMs).toBeCloseTo(2000, 0)
  })

  it('manual mode clicks at the song tempo to begin with', async () => {
    render(<Harness />)
    await enable()
    expect(screen.getByRole('textbox', { name: 'BPM' })).toHaveValue('120')
    const beats = latest().source.beatsBetween(0, 1100)
    expect(beats.map((b) => b.ms)).toEqual([0, 500, 1000])
    expect(beats.map((b) => b.accent)).toEqual([true, false, false])
  })
})

describe('song sync', () => {
  it('follows the beats of the grid, with the first beat of each bar accented', async () => {
    render(<Harness />)
    await enable()
    await userEvent.click(screen.getByRole('radio', { name: 'Đồng bộ bài' }))
    const beats = latest().source.beatsBetween(0, 3000)
    expect(beats[0]).toEqual({ ms: 500, accent: true })
    expect(beats.slice(0, 5).map((b) => b.accent)).toEqual([true, false, false, false, true])
    expect(beats.map((b) => b.ms)).toEqual(grid.bars.flatMap((bar) => bar.beatsMs).filter((ms) => ms < 3000))
  })

  it('joins the song position, the audio clock and the current speed', async () => {
    const { rerender } = render(<Harness rate={0.75} />)
    await enable()
    await userEvent.click(screen.getByRole('radio', { name: 'Đồng bộ bài' }))
    expect(latest().clock.sample()).toEqual({ songMs: 4321, ctxTime: 12.5, rate: 0.75, playing: false })
    transport.state = 'playing'
    transport.pos = 9000
    ctx.currentTime = 13
    rerender(<Harness rate={0.5} />)
    expect(latest().clock.sample()).toEqual({ songMs: 9000, ctxTime: 13, rate: 0.5, playing: true })
  })
})

describe('keeping in step', () => {
  it('forgets its schedule when the speed changes', async () => {
    const { rerender } = render(<Harness />)
    await enable()
    const scheduler = latest()
    rerender(<Harness rate={0.75} />)
    expect(scheduler.reset).toHaveBeenCalled()
  })

  it('forgets its schedule on a seek and on a loop wrap', async () => {
    render(<Harness />)
    await enable()
    const scheduler = latest()
    act(() => controller.seek(8000))
    expect(scheduler.reset).toHaveBeenCalledTimes(1)
    controller.loop.setFromBars(grid, 2, 3)
    act(() => {
      controller.frame(grid.barEndMs(3) - 10)
      controller.frame(grid.barEndMs(3) + 10)
    })
    expect(scheduler.reset.mock.calls.length).toBeGreaterThanOrEqual(2)
    controller.loop.clear()
  })

  it('stops when switched off and when the page is left', async () => {
    const { unmount } = render(<Harness />)
    await enable()
    const scheduler = latest()
    await enable()
    expect(scheduler.stop).toHaveBeenCalledTimes(1)
    await enable()
    const second = latest()
    unmount()
    expect(second.stop).toHaveBeenCalled()
  })
})

describe('tempo', () => {
  const bpm = () => screen.getByRole('textbox', { name: 'BPM' })
  const typeBpm = async (text: string) => {
    await userEvent.clear(bpm())
    await userEvent.type(bpm(), text)
    await userEvent.tab()
  }

  it.each([
    ['10', '30'],
    ['999', '280'],
    ['90', '90'],
    ['abc', '120'],
  ])('turns %s into %s', async (typed, shown) => {
    render(<Harness />)
    await typeBpm(typed)
    expect(bpm()).toHaveValue(shown)
  })

  it('takes a tempo from Tap Tempo', async () => {
    render(<Harness />)
    const now = vi.spyOn(performance, 'now')
    const tap = screen.getByRole('button', { name: 'Tap' })
    for (const t of [1000, 1500, 2000]) {
      now.mockReturnValue(t)
      fireEvent.click(tap)
    }
    expect(bpm()).toHaveValue('120')
    now.mockRestore()
  })

  it('rebuilds the clicks when the tempo changes while running', async () => {
    render(<Harness />)
    await enable()
    await typeBpm('60')
    expect(latest().source.beatsBetween(0, 1100).map((b) => b.ms)).toEqual([0, 1000])
  })
})

describe('sound', () => {
  it('passes the timbre and the volume to the running scheduler', async () => {
    render(<Harness />)
    await enable()
    const scheduler = latest()
    await userEvent.selectOptions(screen.getByLabelText('Âm sắc'), 'beep')
    expect(scheduler.setTimbre).toHaveBeenLastCalledWith('beep')
    fireEvent.change(screen.getByLabelText('Âm lượng'), { target: { value: '0.3' } })
    expect(scheduler.setVolume).toHaveBeenLastCalledWith(0.3)
  })
})

describe('count-in', () => {
  it('is off by default: playing starts at once', async () => {
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'go' }))
    expect(scheduleClick).not.toHaveBeenCalled()
    expect(transport.play).toHaveBeenCalledTimes(1)
  })

  it('plays one bar of clicks, the first accented, then starts the song', async () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('checkbox', { name: 'Đếm nhịp vào bài' }))
    vi.useFakeTimers()
    fireEvent.click(screen.getByRole('button', { name: 'go' }))
    await act(async () => {})
    const calls = vi.mocked(scheduleClick).mock.calls
    expect(calls).toHaveLength(4)
    expect(calls.map(([, when]) => Number((when - calls[0][1]).toFixed(3)))).toEqual([0, 0.5, 1, 1.5])
    expect(calls.map(([, , opts]) => opts.accent)).toEqual([true, false, false, false])
    expect(transport.play).not.toHaveBeenCalled()
    await act(async () => {
      vi.advanceTimersByTime(1900)
    })
    expect(transport.play).not.toHaveBeenCalled()
    await act(async () => {
      vi.advanceTimersByTime(300)
    })
    expect(transport.play).toHaveBeenCalledTimes(1)
  })

  it('is not played if the page is left first', async () => {
    const { unmount } = render(<Harness />)
    fireEvent.click(screen.getByRole('checkbox', { name: 'Đếm nhịp vào bài' }))
    vi.useFakeTimers()
    fireEvent.click(screen.getByRole('button', { name: 'go' }))
    await act(async () => {})
    unmount()
    await act(async () => {
      vi.advanceTimersByTime(5000)
    })
    expect(transport.play).not.toHaveBeenCalled()
  })
})
