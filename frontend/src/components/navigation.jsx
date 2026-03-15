import { Link, NavLink } from 'react-router-dom'
import { Database, Home } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ThemeSwitch } from '@/components/theme-switch'

export function Navigation() {
  const links = [
    { href: '/', label: 'Overview', icon: Home },
    { href: '/dashboard', label: 'Dashboard', icon: Database },
  ]

  return (
    <nav className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6 sm:px-8 lg:px-10">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md border border-accent/20 bg-accent/10">
              <Database className="h-5 w-5 text-accent" />
            </div>
            <span className="text-lg font-semibold">DataFlow</span>
          </Link>
          <div className="hidden md:flex md:gap-1">
            {links.map((link) => {
              const Icon = link.icon
              return (
                <NavLink
                  key={link.href}
                  to={link.href}
                  className={({ isActive }) =>
                    cn(
                      'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
                    )
                  }
                >
                  <Icon className="h-4 w-4" />
                  {link.label}
                </NavLink>
              )
            })}
          </div>
        </div>
        <ThemeSwitch />
      </div>
    </nav>
  )
}
