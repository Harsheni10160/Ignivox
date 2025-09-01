"use client"

import Link from "next/link"
import { HomeCard } from "@/components/home-card"
import { Button } from "@/components/ui/button"
import { BottomTabs } from "@/components/bottom-tabs"
import { useRouter } from "next/navigation"
import { useAuth } from "@/context/auth-context"

import { Send, Mic, Camera, QrCode } from "lucide-react"

export default function HomePage() {
  const router = useRouter()
  const { user } = useAuth()

  return (
    <main className="mx-auto mb-20 flex min-h-dvh max-w-md flex-col gap-4 p-4">
      {!user ? (
        <div className="rounded-xl border p-4">
          <h1 className="text-lg font-semibold text-balance">Welcome to IGNIVOX</h1>
          <p className="mt-1 text-sm text-muted-foreground">Please log in to continue.</p>
          <Button className="mt-3 w-full" onClick={() => router.push("/login")}>
            Log in
          </Button>
        </div>
      ) : (
        <>
          <HomeCard />
          <section aria-labelledby="quick-pay-heading">
            <h2 id="quick-pay-heading" className="mb-2 text-sm font-medium">
              Quick Pay
            </h2>
            <div className="grid grid-cols-2 gap-3">
              {/* Send Money */}
              <Link
                href="/pay"
                className="group rounded-xl border p-4 transition-colors hover:bg-foreground hover:text-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
              >
                <Send className="h-6 w-6" aria-hidden />
                <div className="mt-2 font-semibold">Send Money</div>
              </Link>

              {/* Speak to Pay */}
              <Link
                href="/pay?speak=1"
                className="group rounded-xl border p-4 transition-colors hover:bg-foreground hover:text-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
              >
                <Mic className="h-6 w-6" aria-hidden />
                <div className="mt-2 font-semibold">Speak to Pay</div>
              </Link>

              {/* Face Pay */}
              <Link
                href="/pay?face=1"
                className="group rounded-xl border p-4 transition-colors hover:bg-foreground hover:text-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
              >
                <Camera className="h-6 w-6" aria-hidden />
                <div className="mt-2 font-semibold">Face Pay</div>
              </Link>

              {/* Scan QR */}
              <Link
                href="/pay?qr=1"
                className="group rounded-xl border p-4 transition-colors hover:bg-foreground hover:text-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
              >
                <QrCode className="h-6 w-6" aria-hidden />
                <div className="mt-2 font-semibold">Scan QR</div>
              </Link>
            </div>
          </section>
        </>
      )}
      <BottomTabs />
    </main>
  )
}
