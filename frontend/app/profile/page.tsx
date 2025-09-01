"use client"

import { useAuth } from "@/context/auth-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { BottomTabs } from "@/components/bottom-tabs"

export default function ProfilePage() {
  const {
    user,
    logout,
    securityPin,
    setSecurityPin,
    isDark,
    toggleDark,
    language,
    setLanguage,
    deviceLock,
    setDeviceLock,
  } = useAuth()

  return (
    <main className="mx-auto mb-20 max-w-md p-4">
      <h1 className="mb-3 text-lg font-semibold">Profile</h1>

      <section className="rounded-xl border p-4">
        <h2 className="mb-2 text-sm font-medium">Account</h2>
        <div className="text-sm">
          <div>Name: {user?.name}</div>
          <div>Email: {user?.email}</div>
        </div>
        <Button className="mt-3 w-full bg-transparent" variant="outline" onClick={logout}>
          Log out
        </Button>
      </section>

      <section className="mt-4 rounded-xl border p-4">
        <h2 className="mb-2 text-sm font-medium">Security</h2>
        <Label htmlFor="pin">Security PIN</Label>
        <div className="mt-1 flex gap-2">
          <Input
            id="pin"
            type="password"
            inputMode="numeric"
            pattern="[0-9]*"
            value={securityPin}
            onChange={(e) => setSecurityPin(e.target.value)}
            className="max-w-[200px]"
          />
        </div>
        <div className="mt-3 flex items-center justify-between">
          <span className="text-sm">Device Lock</span>
          <button
            className="rounded-full border px-3 py-1 text-sm"
            onClick={() => setDeviceLock(!deviceLock)}
            aria-pressed={deviceLock}
          >
            {deviceLock ? "On" : "Off"}
          </button>
        </div>
      </section>

      <section className="mt-4 rounded-xl border p-4">
        <h2 className="mb-2 text-sm font-medium">Preferences</h2>
        <div className="mb-3 flex items-center justify-between">
          <span className="text-sm">Dark Mode</span>
          <button className="rounded-full border px-3 py-1 text-sm" onClick={toggleDark} aria-pressed={isDark}>
            {isDark ? "On" : "Off"}
          </button>
        </div>
        <div>
          <Label htmlFor="lang">Language</Label>
          <select
            id="lang"
            className="mt-1 w-full rounded-md border bg-background p-2"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="ta">Tamil</option>
            <option value="te">Telugu</option>
            <option value="kn">Kannada</option>
            <option value="bn">Bengali</option>
          </select>
        </div>
      </section>

      <BottomTabs />
    </main>
  )
}
