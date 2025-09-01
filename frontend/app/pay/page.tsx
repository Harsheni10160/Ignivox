"use client"

import type React from "react"

import { useEffect, useMemo, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { contacts } from "@/lib/data"
import { VoiceToPay } from "@/components/voice-to-pay"
import { FacePay } from "@/components/face-pay"
import { QrScanner } from "@/components/qr-scanner"
import { BottomTabs } from "@/components/bottom-tabs"

export default function PayPage() {
  const sp = useSearchParams()
  const router = useRouter()

  const [to, setTo] = useState<string>("")
  const [amount, setAmount] = useState<string>("")
  const [note, setNote] = useState<string>("")
  const method = sp.get("method") || undefined

  const incomingAmount = sp.get("amount") || ""
  const incomingToId = sp.get("to") || ""
  const incomingToName = sp.get("toName") || ""
  const showSpeak = sp.get("speak") === "1"
  const showFace = sp.get("face") === "1"
  const showQr = sp.get("qr") === "1"

  useEffect(() => {
    if (incomingAmount) setAmount(incomingAmount)
    if (incomingToId) setTo(incomingToId)
    if (incomingToName) setTo(incomingToName)
  }, [incomingAmount, incomingToId, incomingToName])

  const toOptions = useMemo(() => contacts, [])

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const amt = Number.parseInt(amount, 10)
    if (!amt || !to) return
    const found = contacts.find((c) => c.id === to || c.name === to)
    const toName = found?.name || to
    router.push(
      `/pay/confirm?amount=${amt}&toName=${encodeURIComponent(toName)}&method=${method || "manual"}&note=${encodeURIComponent(note)}`,
    )
  }

  return (
    <main className="mx-auto mb-20 max-w-md p-4">
      <h1 className="mb-3 text-lg font-semibold">Send Money</h1>

      {showSpeak && (
        <div className="mb-4 rounded-xl border p-3">
          <h2 className="mb-2 text-sm font-medium">Speak to Pay</h2>
          <VoiceToPay />
        </div>
      )}

      {showFace && (
        <div className="mb-4 rounded-xl border p-3">
          <h2 className="mb-2 text-sm font-medium">Face Pay</h2>
          <FacePay />
        </div>
      )}

      {showQr && (
        <div className="mb-4 rounded-xl border p-3">
          <h2 className="mb-2 text-sm font-medium">Scan QR</h2>
          <QrScanner />
        </div>
      )}

      <form onSubmit={onSubmit} className="rounded-xl border p-4">
        <div className="mb-3">
          <Label htmlFor="to">Recipient</Label>
          <select
            id="to"
            className="mt-1 w-full rounded-md border bg-background p-2"
            value={to}
            onChange={(e) => setTo(e.target.value)}
          >
            <option value="">Select contact</option>
            {toOptions.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} ({c.upiId})
              </option>
            ))}
            {incomingToName && <option value={incomingToName}>{incomingToName} (from QR)</option>}
          </select>
        </div>

        <div className="mb-3">
          <Label htmlFor="amount">Amount (₹)</Label>
          <Input
            id="amount"
            inputMode="numeric"
            pattern="[0-9]*"
            placeholder="500"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
        </div>

        <div className="mb-3">
          <Label htmlFor="note">Note (optional)</Label>
          <Input id="note" placeholder="For lunch, rent, etc." value={note} onChange={(e) => setNote(e.target.value)} />
        </div>

        <Button type="submit" className="w-full">
          Review Payment
        </Button>
      </form>

      <BottomTabs />
    </main>
  )
}
