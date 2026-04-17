import { useEffect } from 'react'

export function useKeyboardShortcuts(shortcuts) {
  useEffect(() => {
    const handleKeyDown = (event) => {
      const target = event.target
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return
      }

      shortcuts.forEach(({ key, ctrl, shift, callback }) => {
        const ctrlPressed = ctrl ? event.ctrlKey : true
        const shiftPressed = shift ? event.shiftKey : !event.shiftKey
        
        if (event.key === key && ctrlPressed && shiftPressed) {
          event.preventDefault()
          callback()
        }
      })
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [shortcuts])
}
