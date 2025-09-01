"use client"

import { useState } from "react"
import { IgnivoxLogo } from "./logo-ignivox"
import { PasswordGate } from "./password-gate"

export function HomeCard() {
  const [revealed, setRevealed] = useState(false)

  return (
    <div className="rounded-2xl border p-4">
      <div className="flex items-start justify-between">
        <div className="flex flex-col">
          <IgnivoxLogo />
          <p className="text-xs text-muted-foreground">Secure Multi-Modal Payments</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
          <span aria-hidden>✓</span> Voice + Face Active
        </div>
      </div>

      <div className="mt-4 rounded-xl bg-muted/40 p-3">
        <div className="flex items-center justify-between">
          <span className="text-sm">Total Balance</span>
          <div className="flex items-center gap-2">
            {!revealed ? (
              <span className="font-semibold tracking-widest">₹ ••••••</span>
            ) : (
              <span className="font-semibold">₹ 25,430.75</span>
            )}
            {!revealed ? (
              <PasswordGate onUnlocked={() => setRevealed(true)} />
            ) : (
              <button
                className="text-xs text-muted-foreground"
                onClick={() => setRevealed(false)}
                aria-label="Hide balance"
              >
                Hide
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
