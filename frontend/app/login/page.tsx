"use client"
import React, { useRef, useState } from "react"
import { useAuth } from "@/context/auth-context"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

const STATEMENT = "I am logging in to IGNIVOX and authorizing my identity."

export default function LoginPage() {
  const { login } = useAuth()
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [photoData, setPhotoData] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  // Webcam
  const videoRef = useRef<HTMLVideoElement | null>(null)
  // Audio
  const [recording, setRecording] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  // Start webcam
  React.useEffect(() => {
    if (!videoRef.current) return
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      })
      .catch(() => setError("Could not access webcam."))
  }, [videoRef.current])

  // Take photo from webcam
  const capturePhoto = () => {
    if (!videoRef.current) return
    const video = videoRef.current
    const canvas = document.createElement("canvas")
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height)
    const dataUrl = canvas.toDataURL("image/png")
    setPhotoData(dataUrl)
  }

  // Voice recording
  const handleStartRecording = async () => {
    setRecording(true)
    setAudioBlob(null)
    setAudioUrl(null)
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const mediaRecorder = new MediaRecorder(stream)
    mediaRecorderRef.current = mediaRecorder
    const audioChunks: BlobPart[] = []
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data)
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" })
      setAudioBlob(audioBlob)
      setAudioUrl(URL.createObjectURL(audioBlob))
      stream.getTracks().forEach(track => track.stop())
      setRecording(false)
    }
    mediaRecorder.start()
  }
  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
    }
  }

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError("")
    if (!photoData || !audioBlob) {
      setError("Please take a live photo and record your voice.")
      setLoading(false)
      return
    }
    // Convert data URL to File
    const photoFile = await (await fetch(photoData)).blob()
    const formData = new FormData()
    formData.append("email", email)
    formData.append("password", password)
    formData.append("face", photoFile, "photo.png")
    formData.append("voice", audioBlob, "voice.webm")
    // Send to backend here if needed
    // const res = await fetch("/api/login", { method: "POST", body: formData })
    // const ok = await res.json()
    const ok = await login(email, password)
    setLoading(false)
    if (!ok) {
      setError("Invalid credentials or biometrics")
    } else {
      router.replace("/")
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-md items-center p-4">
      <form onSubmit={onSubmit} className="w-full rounded-xl border p-4" encType="multipart/form-data">
        <h1 className="mb-3 text-lg font-semibold">Log in to IGNIVOX</h1>
        <Label htmlFor="email">Email</Label>
        <Input id="email" type="email" className="mb-3" value={email} onChange={e => setEmail(e.target.value)} />
        <Label htmlFor="password">Password</Label>
        <Input id="password" type="password" className="mb-3" value={password} onChange={e => setPassword(e.target.value)} />
        
        <div className="mb-3">
          <Label htmlFor="face-live">Live Face (Webcam)</Label>
          <div>
            <video ref={videoRef} autoPlay muted width={240} height={180} style={{ borderRadius: "0.5rem", border: "1px solid #ccc" }} />
          </div>
          <Button type="button" className="mt-2 mb-2" onClick={capturePhoto}>Take Photo</Button>
          {photoData && <img src={photoData} alt="Captured face" className="mb-2" width={120} />}
        </div>
        
        <div className="mb-3">
          <Label htmlFor="voice">Live Voice Recording</Label>
          <div className="mb-1 font-medium">Read this statement aloud:</div>
          <div className="mb-1 italic">&ldquo;{STATEMENT}&rdquo;</div>
          {!recording && (
            <Button type="button" onClick={handleStartRecording}>Start recording</Button>
          )}
          {recording && (
            <Button type="button" onClick={handleStopRecording}>Stop recording</Button>
          )}
          {audioUrl && (
            <audio src={audioUrl} controls className="block mt-2" />
          )}
        </div>
        
        {error && <p className="mb-2 text-sm text-red-600">{error}</p>}
        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Logging in..." : "Log in"}
        </Button>
      </form>
    </main>
  )
}
