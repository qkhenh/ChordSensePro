import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router'
import { useAuth } from '../auth/context'
import { EMAIL_PATTERN, errorMessage, text } from '../auth/messages'
import { AuthLayout, Field, FormError, SubmitButton } from './AuthLayout'

const MIN_PASSWORD_LENGTH = 8

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({})
  const [serverError, setServerError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    const next = {
      email: EMAIL_PATTERN.test(email.trim()) ? undefined : text.invalidEmail,
      password: password.length >= MIN_PASSWORD_LENGTH ? undefined : text.passwordTooShort,
    }
    setErrors(next)
    setServerError(null)
    if (next.email || next.password) return

    setPending(true)
    try {
      await register(email.trim(), password, name.trim() || undefined)
      navigate('/', { replace: true })
    } catch (error) {
      setServerError(errorMessage(error))
      setPending(false)
    }
  }

  return (
    <AuthLayout title={text.createAccount}>
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
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          error={errors.password}
        />
        <Field
          id="name"
          label={text.name}
          autoComplete="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <SubmitButton pending={pending} idle={text.register} busy={text.registering} />
      </form>
      <p className="mt-4 text-center text-sm text-slate-400">
        {text.haveAccount}{' '}
        <Link to="/login" className="text-brand-cyan-light hover:underline">
          {text.login}
        </Link>
      </p>
    </AuthLayout>
  )
}
