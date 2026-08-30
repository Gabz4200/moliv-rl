# Apple Human Interface

## Visual language
Use an Apple Human Interface–inspired visual language: calm, precise, content-led, and familiar. Controls should feel native and immediately understandable. This file controls appearance, not page structure.

## Color and materials
- Use semantic system-like colors that adapt cleanly to light and dark mode.
- Build hierarchy with grouped backgrounds, separators, and restrained translucent material. Do not apply glass, blur, or gradients to every surface.
- Keep body text, secondary text, separators, selection, success, warning, and destructive states distinct at WCAG AA contrast.

## Typography
- Use -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", and a system sans-serif fallback.
- Use larger display text sparingly. Most interface text should use 13–17px sizes with compact, readable line height.
- Use weight, alignment, and whitespace before adding color. Use tabular numerals for changing or comparable values.

## Shape and depth
- Use continuous-feeling rounded rectangles: 8–10px for controls and 12–16px for grouped surfaces.
- Prefer hairline separators and subtle material changes over visible card borders.
- Shadows are soft and rare, reserved for menus, sheets, and temporary floating layers.

## Layout rhythm
- Keep the content hierarchy obvious with generous outer margins and compact internal spacing.
- Align labels, values, and controls precisely. Preserve breathing room around the primary content.
- Keep pointer and touch targets at least 44px and account for safe-area insets on full-screen layouts.

## Components
- Use familiar labels and symbols. Icon-only controls require accessible names and should not replace a clearer text action.
- Primary actions are visually clear without becoming oversized. Destructive actions stay explicit and separated from routine actions.
- Use sheets, popovers, and dialogs only for temporary decisions; keep the underlying context visible when useful.

## Motion
- Use quick, natural transitions that reinforce continuity and direct manipulation.
- Avoid decorative looping animation and exaggerated bounce.
- Respect prefers-reduced-motion and keep every action usable without motion.
