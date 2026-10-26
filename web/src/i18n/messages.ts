import type { Quota } from '../api/types'
import type { FileProblem, UrlProblem } from '../analyze/validate'

export type ProblemKey = FileProblem | UrlProblem

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
  analyze: {
    tabs: { label: 'Nguồn nhạc', upload: 'Tải lên', record: 'Ghi âm', link: 'Liên kết' },
    recordSoon: 'Ghi âm trực tiếp sẽ sớm có mặt',
    dropPrompt: 'Kéo thả tệp nhạc vào đây hoặc',
    chooseFile: 'chọn tệp',
    formats: 'MP3, WAV, FLAC, M4A hoặc OGG · tối đa 100 MB và 10 phút',
    linkLabel: 'Dán liên kết YouTube, SoundCloud hoặc tệp âm thanh',
    submit: 'Phân tích bài hát',
    sending: 'Đang gửi…',
    uploadInstead: 'Tải tệp lên thay thế',
    problems: {
      fileEmpty: 'Tệp trống',
      fileTooLarge: 'Tệp lớn hơn 100 MB',
      unsupportedFormat: 'Chỉ nhận tệp MP3, WAV, FLAC, M4A hoặc OGG',
      urlInvalid: 'Liên kết không hợp lệ',
      urlScheme: 'Chỉ nhận liên kết http hoặc https',
      urlCredentials: 'Liên kết không được chứa tên đăng nhập hoặc mật khẩu',
      urlPrivate: 'Không nhận địa chỉ nội bộ hoặc localhost',
      urlPort: 'Chỉ nhận cổng 80 hoặc 443',
    } satisfies Record<ProblemKey, string>,
  },
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
  analyze: {
    tabs: { label: 'Music source', upload: 'Upload', record: 'Record', link: 'Link' },
    recordSoon: 'Recording from the microphone is coming soon',
    dropPrompt: 'Drop a music file here or',
    chooseFile: 'choose a file',
    formats: 'MP3, WAV, FLAC, M4A or OGG · up to 100 MB and 10 minutes',
    linkLabel: 'Paste a YouTube, SoundCloud or audio file link',
    submit: 'Analyze song',
    sending: 'Sending…',
    uploadInstead: 'Upload a file instead',
    problems: {
      fileEmpty: 'The file is empty',
      fileTooLarge: 'The file is larger than 100 MB',
      unsupportedFormat: 'Only MP3, WAV, FLAC, M4A or OGG files are accepted',
      urlInvalid: 'This is not a valid link',
      urlScheme: 'Only http or https links are accepted',
      urlCredentials: 'Links with a user name or password are not accepted',
      urlPrivate: 'Internal addresses and localhost are not accepted',
      urlPort: 'Only ports 80 and 443 are accepted',
    },
  },
}

export const dictionaries: Record<Lang, Messages> = { vi, en }

/** Endonyms: the switch reads the same whichever language is active. */
export const LANG_NAMES: Record<Lang, string> = { vi: 'Tiếng Việt', en: 'English' }
