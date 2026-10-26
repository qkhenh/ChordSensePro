import { Navigate, type RouteObject } from 'react-router'
import type { ApiClient } from './api/http'
import { GuestOnly, RequireAuth } from './auth/guards'
import { Providers } from './Providers'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import SectionPage from './pages/SectionPage'
import WorkspacePage from './pages/WorkspacePage'
import { ShellLayout } from './shell/ShellLayout'

export function buildRoutes(client: ApiClient): RouteObject[] {
  return [
    {
      element: <Providers client={client} />,
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
            {
              element: <ShellLayout />,
              children: [
                { path: '/', element: <Navigate to="/workspace" replace /> },
                { path: '/library', element: <SectionPage section="library" /> },
                { path: '/workspace', element: <WorkspacePage /> },
                { path: '/dictionary', element: <SectionPage section="dictionary" /> },
              ],
            },
          ],
        },
      ],
    },
  ]
}
