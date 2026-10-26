import { toastFor } from '../api/errors'

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

/** Friendly Vietnamese text for an API failure. */
export const errorMessage = (error: unknown): string => toastFor(error).message

export const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
