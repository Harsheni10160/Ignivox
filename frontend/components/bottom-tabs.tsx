"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Home, Send, History, User } from "lucide-react"

const tabs = [
  { href: "/", label: "Home", Icon: Home },
  { href: "/pay", label: "Pay", Icon: Send },
  { href: "/history", label: "History", Icon: History },
  { href: "/profile", label: "Profile", Icon: User },
]

export function BottomTabs() {
  const pathname = usePathname()
  return (
    <nav className="fixed bottom-0 left-0 right-0 border-t bg-background">
      <div className="mx-auto flex max-w-md items-stretch justify-between px-4">
        {tabs.map(({ href, label, Icon }) => {
          const active = pathname === href
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex flex-1 flex-col items-center justify-center py-2 text-xs transition-colors",
                active ? "bg-foreground text-background font-semibold" : "text-muted-foreground hover:text-foreground",
              )}
              aria-current={active ? "page" : undefined}
            >
              <Icon className="h-5 w-5" aria-hidden />
              <span className="mt-1">{label}</span>
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
