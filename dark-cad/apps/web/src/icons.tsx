import type { JSX } from 'solid-js'

// 24×24 stroke icons (Lucide-style). One path string (or several, space-separated "M" runs) per icon.
const PATHS = {
  sketch: 'M4 20l4-1 11-11a2.1 2.1 0 00-3-3L5 16l-1 4z M14 7l3 3',
  select: 'M5 3l14 7-6 2-2 6z',
  rect: 'M4 6h16v12H4z',
  circle: 'M12 4a8 8 0 100 16 8 8 0 000-16z',
  polygon: 'M12 3l8 6-3 11H7L4 9z',
  extrude: 'M12 3l8 4.5v9L12 21l-8-4.5v-9z M4 7.5l8 4.5 8-4.5 M12 12v9',
  loft: 'M7 5h10 M4 19h16 M7 5l-3 14 M17 5l3 14',
  combine: 'M3 3h11v11H3z M10 10h11v11H10z',
  thread: 'M9 3h6v3H9z M7 8l10 3 M7 12l10 3 M7 16l10 3',
  deleteFeature: 'M4 7h16 M9 7V4h6v3 M6 7l1 13h10l1-13 M10 11v6 M14 11v6',
  newFile: 'M6 3h8l4 4v14H6z M14 3v4h4',
  open: 'M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z',
  save: 'M5 3h11l3 3v15H5z M8 3v5h7V3 M8 21v-7h8v7',
  import: 'M12 3v12 M7 10l5 5 5-5 M4 21h16',
  export: 'M12 15V3 M7 8l5-5 5 5 M4 21h16',
  undo: 'M9 14L4 9l5-5 M4 9h10a6 6 0 010 12h-3',
  redo: 'M15 14l5-5-5-5 M20 9H10a6 6 0 000 12h3',
  fit: 'M4 9V4h5 M20 9V4h-5 M4 15v5h5 M20 15v5h-5',
  chevron: 'M9 6l6 6-6 6',
  cube: 'M12 3l8 4.5v9L12 21l-8-4.5v-9z M4 7.5l8 4.5 8-4.5 M12 12v9',
  folder: 'M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z',
  close: 'M6 6l12 12 M18 6L6 18',
} as const

export type IconName = keyof typeof PATHS

export function Icon(props: { name: IconName; size?: number }): JSX.Element {
  return (
    <svg width={props.size ?? 18} height={props.size ?? 18} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d={PATHS[props.name]} />
    </svg>
  )
}
