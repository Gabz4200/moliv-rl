# shadcn/ui

## Visual language
Use the crisp, neutral visual language associated with shadcn/ui: semantic tokens, strong component states, thin borders, restrained radius, and excellent form ergonomics. Recreate the visual principles with ordinary CSS; do not assume Tailwind or shadcn components are installed. This file controls appearance, not page structure.

## Tokens and color
- Define light and dark semantic tokens for background, foreground, card, popover, primary, secondary, muted, accent, destructive, border, input, and ring.
- Use background/foreground pairs so text and icons always match their surface.
- Start from a neutral, zinc, stone, or slate base and add one purposeful brand accent. Keep charts and status colors distinct and accessible.

## Typography
- Use a modern system sans-serif such as Inter or Geist when available, then a system fallback.
- Keep interface text compact: 12px labels, 14–16px body, and 20–30px section headings.
- Use medium and semibold weights for control hierarchy and tabular numerals for aligned values.

## Shape and borders
- Use a shared radius token around 10px, deriving smaller radii for fields and larger radii for dialogs.
- Use 1px borders and visible focus rings. Prefer borders and surface contrast over heavy shadows.
- Do not put every text block inside a card; a card must express a real group or interaction boundary.

## Layout rhythm
- Use a disciplined 4px spacing scale and responsive grid or flex layouts.
- Keep forms compact but never reduce touch targets below 44px on mobile.
- Place labels, descriptions, errors, and actions consistently so users can scan repeated controls.

## Components
- Give buttons, inputs, selects, tabs, tables, dialogs, popovers, tooltips, and toasts consistent hover, active, disabled, and focus-visible states.
- Use explicit labels and concise helper text. Keep errors next to their field and preserve user input after failure.
- Tables and data rows use aligned columns, quiet dividers, and responsive alternatives rather than horizontal overflow.

## Motion
- Use 120–200ms opacity and transform transitions for popovers, dialogs, selection, and disclosure.
- Never use transition: all. Keep motion interruptible and honor prefers-reduced-motion.
