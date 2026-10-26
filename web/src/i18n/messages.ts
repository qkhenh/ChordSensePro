import type { Quota } from '../api/types'

export type Lang = 'vi' | 'en'
export const LANGS: readonly Lang[] = ['vi', 'en']

const vi = {
  nav: {
    label: 'Điều hướng chính',
    library: 'Thư viện',
    workspace: 'Workspace',
    dictionary: 'Từ điển hợp âm',
  },
  pages: {
    library: 'Các bài đã phân tích sẽ xuất hiện ở đây.',
    workspace: 'Tải một bài hát lên hoặc dán liên kết để bắt đầu phân tích hợp âm.',
    dictionary: 'Tra cứu thế bấm và nốt của từng hợp âm.',
  },
  header: { logout: 'Đăng xuất', language: 'Ngôn ngữ' },
  tier: { free: 'Free', pro: 'Pro', education: 'Education' } satisfies Record<Quota['tier'], string>,
  quota: {
    left: (remaining: number, limit: number) => `còn ${remaining}/${limit}`,
    unlimited: 'không giới hạn',
    exhausted: 'hết lượt hôm nay',
    unknown: '—',
    resetsAt: (when: string) => `Làm mới lúc ${when}`,
  },
  toast: { close: 'Đóng' },
}

export type Messages = typeof vi

const en: Messages = {
  nav: {
    label: 'Main navigation',
    library: 'Library',
    workspace: 'Workspace',
    dictionary: 'Dictionary',
  },
  pages: {
    library: 'Songs you have analysed will show up here.',
    workspace: 'Upload a song or paste a link to start analysing its chords.',
    dictionary: 'Look up the fingering and notes of any chord.',
  },
  header: { logout: 'Sign out', language: 'Language' },
  tier: { free: 'Free', pro: 'Pro', education: 'Education' },
  quota: {
    left: (remaining, limit) => `${remaining}/${limit} left`,
    unlimited: 'unlimited',
    exhausted: 'no analyses left today',
    unknown: '—',
    resetsAt: (when) => `Resets at ${when}`,
  },
  toast: { close: 'Close' },
}

export const dictionaries: Record<Lang, Messages> = { vi, en }

/** Endonyms: the switch reads the same whichever language is active. */
export const LANG_NAMES: Record<Lang, string> = { vi: 'Tiếng Việt', en: 'English' }
