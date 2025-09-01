"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useAuth } from "@/context/auth-context"

export function PasswordGate({ onUnlocked }: { onUnlocked: () => void }) {
  const [open, setOpen] = useState(false)
  const [pin, setPin] = useState("")
  const [error, setError] = useState("")
  const { securityPin } = useAuth()

  const verify = () => {
    if (pin === securityPin) {
      setError("")
      setOpen(false)
      onUnlocked()
    } else {
      setError("Incorrect PIN")
    }
  }

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        className="px-2 text-xs"
        onClick={() => setOpen(true)}
        aria-label="Reveal balance (requires PIN)"
      >
        Reveal
      </Button>

      {open && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
        >
          <div className="w-full max-w-sm rounded-lg bg-background p-4 shadow-lg">
            <h2 className="mb-2 text-center text-base font-semibold">Enter Security PIN</h2>
            <div className="flex gap-2">
              <Input
                type="password"
                inputMode="numeric"
                pattern="[0-9]*"
                placeholder="4-digit PIN"
                value={pin}
                onChange={(e) => setPin(e.target.value)}
                aria-label="PIN input"
              />
              <Button onClick={verify} aria-label="Verify PIN">
                Unlock
              </Button>
            </div>
            {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
            <Button variant="ghost" className="mt-2 w-full" onClick={() => setOpen(false)}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </>
  )
}
