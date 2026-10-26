import { describe, expect, it } from 'vitest'
import { ApiError } from './http'
import { toastFor } from './errors'

const err = (status: number, code: string, message = code) => new ApiError(status, code, message)

describe('toastFor', () => {
  it('turns known codes into Vietnamese text', () => {
    expect(toastFor(err(429, 'quota_exceeded')).message).toMatch(/hết lượt/i)
    expect(toastFor(err(403, 'forbidden')).message).toMatch(/quyền/i)
    expect(toastFor(err(0, 'network_error')).message).toMatch(/kết nối/i)
  })

  it('classifies the toast so the UI can colour it', () => {
    expect(toastFor(err(429, 'quota_exceeded')).kind).toBe('warning')
    expect(toastFor(err(0, 'network_error')).kind).toBe('error')
    expect(toastFor(err(500, 'internal_error')).kind).toBe('error')
  })

  it('shows the first field problem for validation errors', () => {
    const e = err(422, 'validation_error')
    e.fieldErrors.push({ field: 'email', message: 'not an email' })
    expect(toastFor(e).message).toContain('email')
    expect(toastFor(e).message).toContain('not an email')
  })

  it('shows the server message for weak_password, which is actionable', () => {
    expect(toastFor(err(422, 'weak_password', 'Mật khẩu quá ngắn')).message).toBe('Mật khẩu quá ngắn')
  })

  it('quotes the request id on server failures so users can report it', () => {
    const e = err(500, 'internal_error')
    e.requestId = 'req-42'
    expect(toastFor(e).message).toContain('req-42')
  })

  it('falls back to a generic message for unknown codes and non-API errors', () => {
    expect(toastFor(err(418, 'teapot')).message).toBe('Có lỗi xảy ra, vui lòng thử lại')
    expect(toastFor(new Error('boom')).message).toBe('Có lỗi xảy ra, vui lòng thử lại')
    expect(toastFor('weird').kind).toBe('error')
  })
})
