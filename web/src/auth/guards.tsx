import { Navigate, Outlet, useLocation } from 'react-router'
import { useAuth } from './context'
import { destinationFrom } from './destination'
import { text } from './messages'

function Loading() {
  return (
    <p role="status" className="p-8 text-center text-slate-400">
      {text.loading}
    </p>
  )
}

/** Protected routes: anonymous visitors go to /login and come back afterwards. */
export function RequireAuth() {
  const { status } = useAuth()
  const location = useLocation()
  if (status === 'loading') return <Loading />
  if (status === 'anonymous') return <Navigate to="/login" replace state={{ from: location }} />
  return <Outlet />
}

/**
 * Login/register: signed-in users are sent on instead. It must honour the original
 * destination itself: it re-renders the moment the session flips to authenticated,
 * before the login form's own navigate() runs.
 */
export function GuestOnly() {
  const { status } = useAuth()
  const location = useLocation()
  if (status === 'loading') return <Loading />
  if (status === 'authenticated') return <Navigate to={destinationFrom(location.state)} replace />
  return <Outlet />
}
