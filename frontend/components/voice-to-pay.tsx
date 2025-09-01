"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { contacts } from "@/lib/data"

function parseCommand(text: string) {
  const lower = text.toLowerCase()
  const amountMatch = lower.match(/(?:send|pay)\s+(\d+)(?:\s*(?:rupees|₹))?/)
  const toMatch = lower.match(/to\s+([a-z]+(?:\s+[a-z]+)?)/)
  const amount = amountMatch ? Number.parseInt(amountMatch[1], 10) : undefined
  const name = toMatch ? toMatch[1] : undefined
  return { amount, name }
}

export function VoiceToPay() {
  const [listening, setListening] = useState(false)
  const [lastResult, setLastResult] = useState("")
  const [error, setError] = useState("")
  const router = useRouter()
  const recognitionRef = useRef<any>(null)

  const srSupported = useMemo(
    () => typeof window !== "undefined" && ("webkitSpeechRecognition" in window || "SpeechRecognition" in window),
    [],
  )

  useEffect(() => {
    if (!srSupported) return
    const Ctor: any = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    const rec: any = new Ctor()
    rec.lang = "en-IN"
    rec.continuous = false
    rec.interimResults = false
    rec.maxAlternatives = 1
    rec.onresult = (e: any) => {
      const text = e.results[0][0].transcript
      setLastResult(text)
      const { amount, name } = parseCommand(text)
      if (amount && name) {
        const match = contacts.find((c) => c.name.toLowerCase().startsWith(name.toLowerCase()))
        if (match) {
          router.push(`/pay?amount=${amount}&to=${encodeURIComponent(match.id)}&method=voice`)
        } else {
          setError("Could not find contact, please try again.")
        }
      } else if (amount) {
        router.push(`/pay?amount=${amount}&method=voice`)
      } else {
        setError("Sorry, I didn't catch that. Try: 'Send 500 to Ramesh'.")
      }
      setListening(false)
    }
    rec.onerror = (e: any) => {
      setError(e?.error || "Microphone error")
      setListening(false)
    }
    recognitionRef.current = rec
  }, [router, srSupported])

  const start = () => {
    setError("")
    setLastResult("")
    recognitionRef.current?.start()
    setListening(true)
  }

  return (
    <div className="flex w-full flex-col items-center gap-2">
      <Button onClick={start} disabled={!srSupported} className="w-full">
        {listening ? "Listening..." : "Start Voice Command"}
      </Button>
      {!srSupported && <p className="text-sm text-muted-foreground">Voice not supported in this browser.</p>}
      {lastResult && <p className="text-sm text-muted-foreground">Heard: “{lastResult}”</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  )
}
