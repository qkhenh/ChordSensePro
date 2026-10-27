import { Navigate, type RouteObject } from 'react-router'
import type { ApiClient } from './api/http'
import { GuestOnly, RequireAuth } from './auth/guards'
import { Providers } from './Providers'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import PracticePage from './practice/PracticePage'
import LibraryPage from './pages/LibraryPage'
import SectionPage from './pages/SectionPage'
import SyncSpike from './spike/SyncSpike'
import WorkspacePage from './pages/WorkspacePage'
import { ShellLayout } from './shell/ShellLayout'

export function buildRoutes(client: ApiClient): RouteObject[] {
  return [
    // The NFR-01 measurement harness; the dev server only, absent from production builds.
    ...(import.meta.env.DEV ? [{ path: '/__spike/sync', element: <SyncSpike /> }] : []),
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
                { path: '/charts/:id', element: <PracticePage /> },
                { path: '/library', element: <LibraryPage /> },
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
