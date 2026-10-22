import { useState } from 'react'
import { createBrowserRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { apiClient } from './api/http'
import { buildRoutes } from './routes'

export default function App() {
  const [router] = useState(() => createBrowserRouter(buildRoutes(apiClient)))
  return <RouterProvider router={router} />
}
