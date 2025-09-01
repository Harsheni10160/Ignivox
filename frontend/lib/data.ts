import type { Contact, Transaction } from "./types"

export const contacts: Contact[] = [
  { id: "c1", name: "Ramesh Kumar", upiId: "ramesh@upi" },
  { id: "c2", name: "Priya Singh", upiId: "priya@upi" },
  { id: "c3", name: "Rahul Mehta", upiId: "rahul@upi" },
  { id: "c4", name: "Anita Sharma", upiId: "anita@upi" },
]

export const initialTransactions: Transaction[] = [
  {
    id: "t1",
    amount: 500,
    currency: "INR",
    to: "c1",
    toName: "Ramesh Kumar",
    method: "voice",
    status: "success",
    createdAt: new Date(Date.now() - 86400000 * 1).toISOString(),
    note: "Groceries",
  },
  {
    id: "t2",
    amount: 1200,
    currency: "INR",
    to: "c2",
    toName: "Priya Singh",
    method: "qr",
    status: "success",
    createdAt: new Date(Date.now() - 86400000 * 2).toISOString(),
    note: "Rent share",
  },
  {
    id: "t3",
    amount: 150,
    currency: "INR",
    to: "c3",
    toName: "Rahul Mehta",
    method: "manual",
    status: "success",
    createdAt: new Date(Date.now() - 86400000 * 4).toISOString(),
  },
]
