"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import jsQR from "jsqr"
import { useRouter } from "next/navigation"

export function QrScanner() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState("")
  const router = useRouter()

  useEffect(() => {
    let stream: MediaStream | null = null
    let raf = 0

    const tick = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) return
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const code = jsQR(imgData.data, imgData.width, imgData.height)
      if (code?.data) {
        stop()
        handleResult(code.data)
        return
      }
      raf = requestAnimationFrame(tick)
    }

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setActive(true)
        raf = requestAnimationFrame(tick)
      } catch (e: any) {
        setError(e?.message || "Camera access denied")
      }
    }

    const stop = () => {
      stream?.getTracks().forEach((t) => t.stop())
      stream = null
      setActive(false)
      cancelAnimationFrame(raf)
    }

    function handleResult(text: string) {
      try {
        if (text.startsWith("upi://pay")) {
          const url = new URL(text)
          const pa = url.searchParams.get("pa") || ""
          const am = url.searchParams.get("am") || ""
          const name = url.searchParams.get("pn") || pa
          const amount = am ? Number.parseInt(am, 10) : undefined
          const to = encodeURIComponent(name)
          router.push(`/pay?toName=${to}&amount=${amount || ""}&method=qr`)
        } else {
          router.push(`/pay?qr=${encodeURIComponent(text)}&method=qr`)
        }
      } catch {
        router.push(`/pay?qr=${encodeURIComponent(text)}&method=qr`)
      }
    }

    start()
    return () => {
      stop()
    }
  }, [router])

  return (
    <div className="flex flex-col gap-3">
      <div className="relative w-full overflow-hidden rounded-lg border">
        {/* Camera feed */}
        <video ref={videoRef} className="h-64 w-full bg-black object-cover" playsInline />

        {/* Hidden canvas for detection */}
        <canvas ref={canvasRef} className="hidden" />

        <div aria-hidden className="pointer-events-none absolute inset-0">
          {/* Dim background to highlight scan area */}
          <div className="absolute inset-0 bg-black/40" />

          {/* Centered square frame */}
          <div className="absolute left-1/2 top-1/2 h-44 w-44 -translate-x-1/2 -translate-y-1/2 rounded-md border-2 border-white/90 shadow-[0_0_0_9999px_rgba(0,0,0,0.4)]" />

          {/* Corner accents for clarity */}
          <div className="absolute left-1/2 top-1/2 h-44 w-44 -translate-x-1/2 -translate-y-1/2">
            <div className="absolute -left-1 -top-1 h-5 w-5 border-l-2 border-t-2 border-white" />
            <div className="absolute -right-1 -top-1 h-5 w-5 border-r-2 border-t-2 border-white" />
            <div className="absolute -left-1 -bottom-1 h-5 w-5 border-b-2 border-l-2 border-white" />
            <div className="absolute -right-1 -bottom-1 h-5 w-5 border-b-2 border-r-2 border-white" />
          </div>

          {/* Animated scan line */}
          <div className="absolute left-1/2 top-1/2 h-44 w-44 -translate-x-1/2 -translate-y-1/2 overflow-hidden">
            <div
              className="absolute left-0 right-0 h-[2px] bg-white/90"
              style={{ animation: "scanline 2s linear infinite" }}
            />
          </div>
        </div>

        {/* Local CSS keyframes for scan line */}
        <style jsx>{`
          @keyframes scanline {
            0% { transform: translateY(0%); }
            50% { transform: translateY(100%); }
            100% { transform: translateY(0%); }
          }
        `}</style>
      </div>

      {active ? (
        <p className="text-center text-sm text-muted-foreground">Align QR within the frame</p>
      ) : (
        <Button onClick={() => location.reload()} className="w-full">
          Start Scanner
        </Button>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  )
}
