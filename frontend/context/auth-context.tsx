"use client"

import type React from "react"

import { createContext, useContext, useEffect, useMemo, useState } from "react"

type AuthUser = { id: string; name: string; email: string }

type AuthContextValue = {
  user: AuthUser | null
  login: (email: string, password: string) => Promise<boolean>
  logout: () => void
  securityPin: string
  setSecurityPin: (pin: string) => void
  isDark: boolean
  toggleDark: () => void
  language: string
  setLanguage: (lang: string) => void
  deviceLock: boolean
  setDeviceLock: (v: boolean) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [securityPin, setSecurityPin] = useState("1234")
  const [isDark, setIsDark] = useState(false)
  const [language, setLanguage] = useState("en")
  const [deviceLock, setDeviceLock] = useState(true)

  useEffect(() => {
    const u = localStorage.getItem("ignivox:user")
    const pin = localStorage.getItem("ignivox:pin")
    const dark = localStorage.getItem("ignivox:dark")
    const lang = localStorage.getItem("ignivox:lang")
    const dlock = localStorage.getItem("ignivox:deviceLock")
    if (u) setUser(JSON.parse(u))
    if (pin) setSecurityPin(pin)
    if (dark) setIsDark(dark === "1")
    if (lang) setLanguage(lang)
    if (dlock) setDeviceLock(dlock === "1")
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark)
    localStorage.setItem("ignivox:dark", isDark ? "1" : "0")
  }, [isDark])

  const login = async (email: string, password: string) => {
    if (password.length < 4) return false
    const u = { id: "u1", name: "Ignivox User", email }
    setUser(u)
    localStorage.setItem("ignivox:user", JSON.stringify(u))
    return true
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem("ignivox:user")
  }

  const value = useMemo(
    () => ({
      user,
      login,
      logout,
      securityPin,
      setSecurityPin: (pin: string) => {
        setSecurityPin(pin)
        localStorage.setItem("ignivox:pin", pin)
      },
      isDark,
      toggleDark: () => setIsDark((v) => !v),
      language,
      setLanguage: (l: string) => {
        setLanguage(l)
        localStorage.setItem("ignivox:lang", l)
      },
      deviceLock,
      setDeviceLock: (v: boolean) => {
        setDeviceLock(v)
        localStorage.setItem("ignivox:deviceLock", v ? "1" : "0")
      },
    }),
    [user, securityPin, isDark, language, deviceLock],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
