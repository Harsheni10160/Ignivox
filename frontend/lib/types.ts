export type Contact = {
  id: string
  name: string
  upiId: string
  avatar?: string
}

export type Transaction = {
  id: string
  amount: number
  currency: string
  to: string
  toName: string
  method: "voice" | "face" | "qr" | "manual"
  status: "success" | "failed" | "pending"
  createdAt: string
  note?: string
}
