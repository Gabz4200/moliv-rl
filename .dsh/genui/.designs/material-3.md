# Material 3

## Visual language
Use Google's Material 3 design language: expressive but systematic, with tonal color, clear hierarchy, purposeful elevation, and controls that feel direct and touch-friendly. This file controls appearance, not page structure.

## Color
- Define semantic light and dark roles for background, surface, surface-container, outline, primary, secondary, tertiary, success, warning, and error.
- Build the interface from tonal surfaces rather than stacking shadows. Reserve the strongest primary color for the main action and current selection.
- Keep text and controls at WCAG AA contrast. Never rely on color alone for state.

## Typography
- Use Roboto when available, then a clean system sans-serif fallback.
- Use a deliberate type scale with distinct display, headline, title, body, label, and numeric roles.
- Prefer medium weight and size changes for hierarchy. Keep body copy at 14–16px with comfortable line height.

## Shape and elevation
- Use a coherent shape scale: 8px for small controls, 12–16px for fields and cards, and 24–28px only for prominent containers.
- Pills are reserved for filters, compact status, and segmented choices.
- Use tonal elevation first. Add soft shadows only when a surface must visibly float above another.

## Layout rhythm
- Use an 8px spacing grid with 4px for tight internal adjustments.
- Keep touch targets at least 44px. Let layouts reflow instead of shrinking controls on narrow screens.
- Use whitespace and surface tone to group content; do not wrap every block in a card.

## Components
- Buttons, fields, sliders, switches, chips, dialogs, and navigation should share the same color, state, shape, and focus conventions.
- Give every interactive control clear hover, pressed, selected, disabled, and keyboard-focus states.
- Keep labels visible and place validation or recovery text next to the affected control.

## Motion
- Use Material-style emphasized easing for meaningful changes in selection, expansion, and shared position.
- Keep transitions brief, interruptible, and limited to transform and opacity when possible.
- Respect prefers-reduced-motion and keep the final state fully understandable without animation.
