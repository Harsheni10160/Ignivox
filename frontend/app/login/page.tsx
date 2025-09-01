"use client"

import type React from "react"

import { useState } from "react"
import { useAuth } from "@/context/auth-context"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export default function LoginPage() {
  const { login } = useAuth()
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const ok = await login(email, password)
    if (!ok) {
      setError("Invalid credentials")
    } else {
      router.replace("/")
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-md items-center p-4">
      <form onSubmit={onSubmit} className="w-full rounded-xl border p-4">
        <h1 className="mb-3 text-lg font-semibold">Log in to IGNIVOX</h1>
        <Label htmlFor="email">Email</Label>
        <Input id="email" type="email" className="mb-3" value={email} onChange={(e) => setEmail(e.target.value)} />
        <Label htmlFor="password">Password</Label>
        <Input
          id="password"
          type="password"
          className="mb-3"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="mb-2 text-sm text-red-600">{error}</p>}
        <Button type="submit" className="w-full">
          Log in
        </Button>
      </form>
    </main>
  )
}
