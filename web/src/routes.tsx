import { Outlet, type RouteObject } from 'react-router'
import type { ApiClient } from './api/http'
import { AuthProvider } from './auth/AuthProvider'
import { GuestOnly, RequireAuth } from './auth/guards'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'

export function buildRoutes(client: ApiClient): RouteObject[] {
  return [
    {
      element: (
        <AuthProvider client={client}>
          <Outlet />
        </AuthProvider>
      ),
      children: [
        {
          element: <GuestOnly />,
          children: [
            { path: '/login', element: <LoginPage /> },
            { path: '/register', element: <RegisterPage /> },
          ],
        },
        {
          element: <RequireAuth />,
          children: [
            { path: '/', element: <HomePage /> },
            { path: '/workspace', element: <HomePage /> },
          ],
        },
      ],
    },
  ]
}
