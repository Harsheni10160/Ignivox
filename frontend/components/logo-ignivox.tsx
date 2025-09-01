export function IgnivoxLogo({ size = 28 }: { size?: number }) {
  return (
    <div className="flex items-center gap-2">
      <svg width={size} height={size} viewBox="0 0 64 64" aria-label="Ignivox logo" role="img">
        <rect width="64" height="64" rx="14" className="fill-black dark:fill-white" />
        <path d="M20 38 L32 10 L44 38 L32 34 Z" className="fill-white dark:fill-black" />
        <circle cx="32" cy="46" r="6" className="fill-white dark:fill-black" />
      </svg>
      <span className="font-semibold tracking-wide">IGNIVOX</span>
    </div>
  )
}
