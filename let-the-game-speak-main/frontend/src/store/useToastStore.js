import { create } from 'zustand'

let toastId = 0

export const useToastStore = create((set) => ({
  toasts: [],
  
  addToast: (toast) => {
    const id = toastId++
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id }]
    }))
    
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter(t => t.id !== id)
      }))
    }, toast.duration || 3000)
    
    return id
  },
  
  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter(t => t.id !== id)
  })),
  
  success: (message) => {
    set((state) => {
      const id = toastId++
      const toast = { id, type: 'success', message }
      setTimeout(() => {
        set((s) => ({ toasts: s.toasts.filter(t => t.id !== id) }))
      }, 3000)
      return { toasts: [...state.toasts, toast] }
    })
  },
  
  error: (message) => {
    set((state) => {
      const id = toastId++
      const toast = { id, type: 'error', message }
      setTimeout(() => {
        set((s) => ({ toasts: s.toasts.filter(t => t.id !== id) }))
      }, 3000)
      return { toasts: [...state.toasts, toast] }
    })
  },
  
  info: (message) => {
    set((state) => {
      const id = toastId++
      const toast = { id, type: 'info', message }
      setTimeout(() => {
        set((s) => ({ toasts: s.toasts.filter(t => t.id !== id) }))
      }, 3000)
      return { toasts: [...state.toasts, toast] }
    })
  }
}))
