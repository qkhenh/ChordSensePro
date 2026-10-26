import { ApiError } from './http'

export interface ToastSpec {
  kind: 'error' | 'warning'
  message: string
}

const GENERIC = 'Có lỗi xảy ra, vui lòng thử lại'

const byCode: Record<string, ToastSpec> = {
  invalid_credentials: { kind: 'error', message: 'Email hoặc mật khẩu không đúng' },
  email_taken: { kind: 'error', message: 'Email này đã được đăng ký' },
  account_disabled: { kind: 'error', message: 'Tài khoản đã bị vô hiệu hoá' },
  network_error: { kind: 'error', message: 'Không kết nối được máy chủ' },
  forbidden: { kind: 'error', message: 'Bạn không có quyền thực hiện thao tác này' },
  quota_exceeded: { kind: 'warning', message: 'Bạn đã hết lượt phân tích hôm nay' },
  not_found: { kind: 'error', message: 'Không tìm thấy nội dung yêu cầu' },
}

/** What to tell the user about an API failure; the server's own text is used only where it is actionable. */
export function toastFor(error: unknown): ToastSpec {
  if (!(error instanceof ApiError)) return { kind: 'error', message: GENERIC }

  if (error.code === 'weak_password') return { kind: 'error', message: error.message }
  if (error.code === 'validation_error') {
    const first = error.fieldErrors[0]
    return { kind: 'error', message: first ? `${first.field}: ${first.message}` : 'Dữ liệu không hợp lệ' }
  }

  const known = byCode[error.code]
  if (known) return known

  if (error.status >= 500 && error.requestId) {
    return { kind: 'error', message: `${GENERIC} (mã yêu cầu: ${error.requestId})` }
  }
  return { kind: 'error', message: GENERIC }
}
