# MuFinder Brand Guidelines

This document defines the visual identity and UI tone for MuFinder.

## Brand Philosophy

MuFinder is an expedition tool for muon-site discovery.
The compass logo is the core metaphor: the muon symbol sits at the center while spin acts as the compass needle.

The product voice should combine:

- explorer confidence and direction
- notebook-like progress and completion
- modern scientific clarity

The interface should feel like a field notebook used for advanced computational science, not a decorative period piece.

## Brand Principles

1. The journey is the interface.
2. Completion should feel earned.
3. Compass motifs guide navigation.
4. Styling stays tactile but practical.
5. Dark mode evokes a night expedition.

## Canonical Logo Asset

Use this file as the canonical app mark:

- src/mufinder/assets/mufinder_logo.png

The runtime logo/icon path should come from:

- src/mufinder/branding.py via logo_path() and app_icon()

Do not introduce alternate canonical logo sources in the UI without updating this guide.

## Color Palette

### Light Mode: Field Notebook

| Token | Hex | Purpose |
| --- | --- | --- |
| bg_primary | #F5EDE0 | Main parchment background |
| bg_surface | #FDF6EC | Card and control surfaces |
| bg_surface_alt | #EDE0CE | Alternating rows and subtle contrast |
| border | #C4AD8F | Warm notebook border tone |
| text_primary | #2C2418 | Primary text |
| text_muted | #7A6B59 | Secondary text |
| accent | #E02020 | Compass red |
| accent_hover | #C61D1D | Hover state |
| accent_pressed | #A71919 | Pressed state |
| seal_gold | #B8860B | Completion seals and milestone marks |

### Dark Mode: Night Expedition

| Token | Hex | Purpose |
| --- | --- | --- |
| bg_primary | #0D1B2A | Main night background |
| bg_surface | #1B2838 | Card and control surfaces |
| bg_surface_alt | #243447 | Alternating rows and subtle contrast |
| border | #3D5068 | Notebook line and divider tone |
| text_primary | #F0E6D3 | Primary text |
| text_muted | #A09484 | Secondary text |
| accent | #E8A838 | Lantern amber |
| accent_hover | #D49530 | Hover state |
| accent_pressed | #C08528 | Pressed state |
| seal_gold | #D4A853 | Completion seals and milestone marks |

Status semantic colors remain centralized in src/mufinder/app/theme.py.

## Typography

Use a dual typographic system:

- Headings: Playfair Display (fallback: Libre Baskerville, Georgia, serif)
- Body and controls: Inter (fallback: Helvetica Neue)

Typography usage:

- Hero and major section headers use the serif stack.
- Interactive controls and dense scientific content use the sans-serif stack.
- Keep copy plain and actionable.

## Workflow Metaphor and Navigation

The workflow should read like an expedition journal.

- Sidebar tree is the journal index.
- Completed stages show a seal icon.
- Current stage shows a compass-needle icon.
- Incomplete stages show an empty dotted marker.
- Home page includes an expedition trail with waypoint states.

Do not use novelty textures or ornamental clutter that harms readability.

## UI Styling Rules

- Use centralized theme tokens from src/mufinder/app/theme.py.
- Avoid hardcoded per-widget colors and one-off inline style strings.
- Keep text labels on transparent backgrounds unless a component requires a local surface.
- Preserve strong contrast in both light and dark themes.
- Keep form inputs and data tables modern and clear; explorer styling should frame workflow context, not obscure data.

## Logo Placement in GUI

Current placements:

- application icon (window, dock, task switcher)
- splash screen
- home dashboard hero area

Any new logo placement should remain subtle and avoid visual clutter.

## Asset Hygiene

- Keep branding assets in docs/branding and src/mufinder/assets.
- Remove duplicate or stale logo variants when they are no longer needed.
- Ensure production UI branding stays aligned with src/mufinder/assets/mufinder_logo.png.
