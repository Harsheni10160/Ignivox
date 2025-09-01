"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { initialTransactions, contacts } from "@/lib/data"
import { Button } from "@/components/ui/button"
import type { Transaction } from "@/lib/types"

export default function HistoryDetail() {
  const { id } = useParams<{ id: string }>()
  const [tx, setTx] = useState<Transaction | null>(null)
  const router = useRouter()

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("ignivox:tx") || "null") as Transaction[] | null
    const list = stored || initialTransactions
    const found = list.find((t) => t.id === id)
    setTx(found || null)
  }, [id])

  if (!tx) {
    return (
      <main className="mx-auto max-w-md p-4">
        <p>Transaction not found.</p>
      </main>
    )
  }

  const contact = contacts.find((c) => c.name === tx.toName)

  return (
    <main className="mx-auto max-w-md p-4">
      <h1 className="mb-3 text-lg font-semibold">Payment Details</h1>
      <div className="rounded-xl border p-4">
        <Item label="Recipient" value={tx.toName} />
        {contact && <Item label="UPI ID" value={contact.upiId} />}
        <Item label="Amount" value={`₹ ${tx.amount}`} />
        <Item label="Method" value={tx.method} />
        <Item label="Status" value={tx.status} />
        <Item label="Date" value={new Date(tx.createdAt).toLocaleString()} />
        {tx.note && <Item label="Note" value={tx.note} />}
      </div>
      <Button className="mt-4 w-full" onClick={() => router.push("/history")}>
        Back to History
      </Button>
    </main>
  )
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div className="mb-2 flex items-center justify-between">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}
