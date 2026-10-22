import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App', () => {
  it('shows the product name and sends an anonymous visitor to the login form', async () => {
    render(<App />)
    expect(await screen.findByRole('heading', { level: 1, name: /chordsense pro/i })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Đăng nhập' })).toBeInTheDocument()
  })
})
