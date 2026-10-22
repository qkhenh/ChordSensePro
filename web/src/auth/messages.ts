import { ApiError } from '../api/http'

export const text = {
  brand: 'ChordSense Pro',
  tagline: 'Phân tích sâu — Luyện tập chuẩn — Thành thạo toàn diện.',
  loading: 'Đang tải…',
  email: 'Email',
  password: 'Mật khẩu',
  name: 'Họ tên (tuỳ chọn)',
  login: 'Đăng nhập',
  loggingIn: 'Đang đăng nhập…',
  register: 'Đăng ký',
  registering: 'Đang đăng ký…',
  createAccount: 'Tạo tài khoản',
  logout: 'Đăng xuất',
  noAccount: 'Chưa có tài khoản?',
  haveAccount: 'Đã có tài khoản?',
  invalidEmail: 'Email không hợp lệ',
  passwordRequired: 'Vui lòng nhập mật khẩu',
  passwordTooShort: 'Mật khẩu cần ít nhất 8 ký tự',
}

const byCode: Record<string, string> = {
  invalid_credentials: 'Email hoặc mật khẩu không đúng',
  email_taken: 'Email này đã được đăng ký',
  account_disabled: 'Tài khoản đã bị vô hiệu hoá',
  network_error: 'Không kết nối được máy chủ',
}

/** Friendly Vietnamese text for an API failure; falls back to the server's message. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return byCode[error.code] ?? (error.code === 'weak_password' ? error.message : 'Có lỗi xảy ra, vui lòng thử lại')
  }
  return 'Có lỗi xảy ra, vui lòng thử lại'
}

export const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
