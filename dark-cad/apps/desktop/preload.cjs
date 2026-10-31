const { contextBridge, ipcRenderer } = require('electron')

// Implements the same Platform port as the browser build (apps/web/src/platform.ts).
contextBridge.exposeInMainWorld('darkcadDesktop', {
  save: (name, data) => ipcRenderer.invoke('save', { name, data }).then(() => undefined),
  open: (accept) => ipcRenderer.invoke('open', accept),
})
