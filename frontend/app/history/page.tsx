"use client"

import Link from "next/link"
import { useEffect, useState } from "react"
import { BottomTabs } from "@/components/bottom-tabs"
import { initialTransactions } from "@/lib/data"
import type { Transaction } from "@/lib/types"

export default function HistoryPage() {
  const [txs, setTxs] = useState<Transaction[]>([])

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("ignivox:tx") || "null")
    setTxs(stored || initialTransactions)
  }, [])

  return (
    <main className="mx-auto mb-20 max-w-md p-4">
      <h1 className="mb-3 text-lg font-semibold">History</h1>
      <div className="space-y-2">
        {txs.map((t) => (
          <Link
            key={t.id}
            href={`/history/${t.id}`}
            className="flex items-center justify-between rounded-xl border p-3"
          >
            <div className="flex flex-col">
              <span className="font-medium">{t.toName}</span>
              <span className="text-xs text-muted-foreground">{new Date(t.createdAt).toLocaleString()}</span>
            </div>
            <div className="text-right">
              <div className="font-semibold">₹ {t.amount}</div>
              <div className="text-xs capitalize text-muted-foreground">{t.method}</div>
            </div>
          </Link>
        ))}
      </div>
      <BottomTabs />
    </main>
  )
}
