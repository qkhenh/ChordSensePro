import { useState, type FormEvent } from 'react'
import { Link, useLocation, useNavigate } from 'react-router'
import { useAuth } from '../auth/context'
import { destinationFrom } from '../auth/destination'
import { EMAIL_PATTERN, errorMessage, text } from '../auth/messages'
import { AuthLayout, Field, FormError, SubmitButton } from './AuthLayout'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const destination = destinationFrom(location.state)

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({})
  const [serverError, setServerError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    const next = {
      email: EMAIL_PATTERN.test(email.trim()) ? undefined : text.invalidEmail,
      password: password ? undefined : text.passwordRequired,
    }
    setErrors(next)
    setServerError(null)
    if (next.email || next.password) return

    setPending(true)
    try {
      await login(email.trim(), password)
      navigate(destination, { replace: true })
    } catch (error) {
      setServerError(errorMessage(error))
      setPending(false)
    }
  }

  return (
    <AuthLayout title={text.login}>
      <form onSubmit={onSubmit} noValidate>
        <FormError message={serverError} />
        <Field
          id="email"
          label={text.email}
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          error={errors.email}
        />
        <Field
          id="password"
          label={text.password}
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          error={errors.password}
        />
        <SubmitButton pending={pending} idle={text.login} busy={text.loggingIn} />
      </form>
      <p className="mt-4 text-center text-sm text-slate-400">
        {text.noAccount}{' '}
        <Link to="/register" className="text-brand-cyan-light hover:underline">
          {text.register}
        </Link>
      </p>
    </AuthLayout>
  )
}
