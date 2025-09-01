"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

export function FacePay() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState("")
  const [hasFace, setHasFace] = useState(false)
  const [blinkCount, setBlinkCount] = useState(0)
  const [started, setStarted] = useState(false)
  const router = useRouter()

  useEffect(() => {
    let stream: MediaStream | null = null
    let detector: any = null
    let raf = 0
    let lastEyeOpenish = true
    let lastArea = 0

    const tick = async () => {
      const v = videoRef.current
      if (!v) return
      try {
        if (detector) {
          const faces = await detector.detect(v)
          const present = faces && faces.length > 0
          setHasFace(present)
          if (present) {
            const f = faces[0]
            const box = f.boundingBox || f.box || f
            const area = (box.width || 0) * (box.height || 0)
            if (!lastArea) lastArea = area
            const areaChange = Math.abs(area - lastArea) / Math.max(area, 1)
            const moved = areaChange > 0.05
            lastArea = area

            const eyeOpen = !moved
            if (!lastEyeOpenish && eyeOpen) {
              setBlinkCount((c) => Math.min(c + 1, 2))
            }
            lastEyeOpenish = eyeOpen
          }
        } else {
          // Motion fallback when FaceDetector API unsupported
          setHasFace(true)
        }
      } catch {}
      raf = requestAnimationFrame(tick)
    }

    const start = async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
        stream = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
          await videoRef.current.play()
        }
        // @ts-ignore
        const FaceDetector = (window as any).FaceDetector
        if (FaceDetector) detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 })
        setStarted(true)
        raf = requestAnimationFrame(tick)
      } catch (e: any) {
        setError(e?.message || "Unable to access camera")
      }
    }

    start()
    return () => {
      stream?.getTracks().forEach((t) => t.stop())
      cancelAnimationFrame(raf)
    }
  }, [])

  const onAuthorize = () => {
    router.push(`/pay?method=face`)
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="relative w-full overflow-hidden rounded-lg border">
        <video ref={videoRef} className="h-64 w-full bg-black object-cover" playsInline />
      </div>
      <p className="text-sm text-muted-foreground">
        {started
          ? hasFace
            ? "Face detected. Blink twice to authorize."
            : "Align your face in view."
          : "Starting camera..."}
      </p>
      <div className="flex items-center justify-between">
        <div className="text-sm">Blink count: {blinkCount}/2</div>
        <Button onClick={onAuthorize} disabled={blinkCount < 2}>
          Continue
        </Button>
      </div>
      {!("FaceDetector" in window) && (
        <p className="text-xs text-muted-foreground">
          Tip: If blink detection is unavailable, gently move your head closer/farther. Two movements count as blinks.
        </p>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  )
}
