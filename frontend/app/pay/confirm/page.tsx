"use client"

import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { useEffect } from "react"

export default function ConfirmPage() {
  const sp = useSearchParams()
  const router = useRouter()
  const amount = sp.get("amount")
  const toName = sp.get("toName")
  const method = sp.get("method") || "manual"
  const note = sp.get("note") || ""

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  const onPay = () => {
    const id = crypto.randomUUID()
    const tx = {
      id,
      amount: Number(amount || 0),
      currency: "INR",
      to: toName || "",
      toName: toName || "",
      method: method as any,
      status: "success",
      createdAt: new Date().toISOString(),
      note,
    }
    const existing = JSON.parse(localStorage.getItem("ignivox:tx") || "[]")
    existing.unshift(tx)
    localStorage.setItem("ignivox:tx", JSON.stringify(existing))
    router.replace(`/history/${id}`)
  }

  return (
    <main className="mx-auto max-w-md p-4">
      <h1 className="mb-3 text-lg font-semibold">Confirm Payment</h1>
      <div className="rounded-xl border p-4">
        <Row label="To" value={toName || ""} />
        <Row label="Amount" value={`₹ ${amount}`} />
        <Row label="Method" value={method} />
        {note && <Row label="Note" value={note} />}
        <Button className="mt-4 w-full" onClick={onPay}>
          Pay
        </Button>
      </div>
    </main>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="mt-2 flex items-center justify-between">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}
