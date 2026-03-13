import { useTheme } from 'next-themes'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

const THEME_OPTIONS = [
  { value: 'dark', label: 'Dark' },
  { value: 'theme-neon', label: 'Neon' },
  { value: 'theme-slate', label: 'Slate' },
  { value: 'theme-cyberpunk', label: 'Cyberpunk' },
  { value: 'theme-pastel', label: 'Pastel' },
]

export function ThemeSwitch({ size = 'sm', className = '' }) {
  const { theme, resolvedTheme, setTheme } = useTheme()

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="text-xs text-muted-foreground">Theme</span>
      <Select value={resolvedTheme || theme || 'light'} onValueChange={setTheme}>
        <SelectTrigger className={size === 'sm' ? 'h-9 w-40 text-xs' : 'w-44'}>
          <SelectValue placeholder="Select theme" />
        </SelectTrigger>
        <SelectContent>
          {THEME_OPTIONS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
