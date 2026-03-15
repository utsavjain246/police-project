import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/globals.css'
import { ThemeProvider } from './components/theme-provider'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      themes={[
        'dark',
        'theme-neon',
        'theme-slate',
        'theme-cyberpunk',
        'theme-pastel',
      ]}
    >
      <App />
    </ThemeProvider>
  </React.StrictMode>,
)
