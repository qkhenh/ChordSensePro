import type { ErrorCode, JobStage, Quota } from '../api/types'
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
    stages: {
      fetching: 'Tải từ liên kết',
      normalizing: 'Chuẩn hoá âm thanh',
      separating: 'Tách nhạc cụ',
      beat_tracking: 'Dò nhịp',
      recognizing: 'Nhận diện hợp âm',
    } satisfies Record<JobStage, string>,
    tracker: {
      steps: 'Các bước phân tích',
      queued: 'Đang xếp hàng',
      running: 'Đang phân tích',
      done: 'Hoàn tất',
      stop: 'Dừng theo dõi',
      close: 'Đóng',
      retry: 'Thử lại',
      loading: 'Đang tải…',
    },
    errors: {
      unsupported_format: 'Tệp không phải âm thanh hợp lệ (MP3, WAV, FLAC, M4A hoặc OGG).',
      file_too_large: 'Tệp lớn hơn 100 MB.',
      duration_too_long: 'Bài nhạc dài hơn 10 phút.',
      url_unresolvable: 'Không lấy được âm thanh từ liên kết này. Hãy thử tải tệp lên.',
      ssrf_blocked: 'Liên kết này không được chấp nhận vì lý do an toàn.',
      quota_exceeded: 'Bạn đã hết lượt phân tích hôm nay.',
      analysis_timeout: 'Phân tích mất quá nhiều thời gian. Lượt phân tích đã được hoàn lại.',
      ai_failure: 'Mô hình không phân tích được bài này. Lượt phân tích đã được hoàn lại.',
      unknown: 'Có lỗi xảy ra khi phân tích, vui lòng thử lại.',
    } satisfies Record<ErrorCode | 'unknown', string>,
    quotaBanner: {
      title: (limit: number | null) =>
        limit === null ? 'Bạn đã hết lượt phân tích hôm nay' : `Bạn đã dùng hết ${limit} lượt phân tích hôm nay`,
      resets: (when: string) => `Lượt mới sẽ có lúc ${when}`,
      upgrade: 'Nâng cấp Pro',
    },
  },
  demo: { heading: 'Thử một bài demo', hint: 'Không tốn lượt phân tích' },
  chart: {
    loading: 'Đang tải bài…',
    notFound: 'Không tìm thấy bài này',
    unavailable: 'Không tải được bài này, vui lòng thử lại',
    back: 'Về Thư viện',
    key: 'Giọng',
    tempo: 'Nhịp độ',
    bpm: 'BPM',
    time: 'Thời điểm',
    chord: 'Hợp âm',
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
    stages: {
      fetching: 'Fetching from the link',
      normalizing: 'Normalizing audio',
      separating: 'Separating instruments',
      beat_tracking: 'Tracking beats',
      recognizing: 'Recognizing chords',
    },
    tracker: {
      steps: 'Analysis steps',
      queued: 'Waiting in the queue',
      running: 'Analyzing',
      done: 'Done',
      stop: 'Stop watching',
      close: 'Close',
      retry: 'Try again',
      loading: 'Loading…',
    },
    errors: {
      unsupported_format: 'This is not a valid audio file (MP3, WAV, FLAC, M4A or OGG).',
      file_too_large: 'The file is larger than 100 MB.',
      duration_too_long: 'The song is longer than 10 minutes.',
      url_unresolvable: 'Could not get audio from this link. Try uploading the file instead.',
      ssrf_blocked: 'This link is not accepted for safety reasons.',
      quota_exceeded: 'You have used all of today\'s analyses.',
      analysis_timeout: 'The analysis took too long. Your analysis has been given back.',
      ai_failure: 'The model could not analyze this song. Your analysis has been given back.',
      unknown: 'Something went wrong during the analysis, please try again.',
    },
    quotaBanner: {
      title: (limit) =>
        limit === null ? "You have used all of today's analyses" : `You have used all ${limit} analyses for today`,
      resets: (when) => `New ones arrive at ${when}`,
      upgrade: 'Upgrade to Pro',
    },
  },
  demo: { heading: 'Try a demo song', hint: 'Costs no analyses' },
  chart: {
    loading: 'Loading the song…',
    notFound: 'This song was not found',
    unavailable: 'Could not load this song, please try again',
    back: 'Back to Library',
    key: 'Key',
    tempo: 'Tempo',
    bpm: 'BPM',
    time: 'Time',
    chord: 'Chord',
  },
}

export const dictionaries: Record<Lang, Messages> = { vi, en }

/** Endonyms: the switch reads the same whichever language is active. */
export const LANG_NAMES: Record<Lang, string> = { vi: 'Tiếng Việt', en: 'English' }
