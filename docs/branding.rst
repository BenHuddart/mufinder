Branding
========

MuFinder presents muon-site discovery as an expedition. The compass logo is the
core metaphor: the muon symbol sits at the centre and its spin is the compass
needle. The interface should read as a field notebook for computational
science: confident about direction, clear about progress, and never a
decorative period piece.

Principles
----------

1. The journey is the interface: the project is a route through workflow
   stages.
2. Completion is earned and marked, never implied.
3. Compass and map motifs guide navigation; they frame the data, not obscure
   it.
4. Styling is tactile but practical. No novelty textures or ornamental clutter
   that harms readability.
5. The light theme is a field notebook; the dark theme is a night expedition.

Logo
----

``src/mufinder/assets/mufinder_logo.png`` is the canonical mark, with
``mufinder_logo.ico`` as its Windows icon. Code reaches them only through
``mufinder.branding``: ``logo_path()`` for the image and ``app_icon()`` for the
platform's application icon. The logo appears as the application icon (window,
dock, task switcher), on the splash screen, and beside the title at the top of
the home page.
A new placement stays subtle; a new logo source is added to
``mufinder.branding``, not referenced directly from a widget.

Colour
------

``mufinder.app.theme`` is the single home for colour. ``light_palette()`` and
``dark_palette()`` define the tokens below, and widgets read them from
``current_palette()`` rather than hard-coding colours or carrying one-off
stylesheets.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Token
     - Light: field notebook
     - Dark: night expedition
   * - ``bg_primary``
     - Parchment main background
     - Deep navy main background
   * - ``bg_surface``, ``bg_surface_alt``
     - Lighter card and control surfaces; alternating rows
     - Raised navy surfaces; alternating rows
   * - ``border``
     - Warm notebook-rule brown
     - Slate divider
   * - ``text_primary``, ``text_muted``
     - Dark ink; muted brown secondary text
     - Warm cream; muted secondary text
   * - ``accent``, ``accent_hover``, ``accent_pressed``
     - Compass red
     - Lantern amber
   * - ``seal_gold``
     - Completion seals and milestone marks
     - Completion seals and milestone marks

Semantic colours (``success``, ``warning``, ``danger``) and the job-state
colours (``status_*``) live in the same palettes. Both themes keep strong
contrast, and text labels sit on transparent backgrounds unless a component
needs its own surface.

Typography
----------

MuFinder bundles its fonts in ``src/mufinder/assets/fonts`` and registers them
with Qt before they are first used, so typography is the same on every
platform:

- Headings use Fraunces (``heading_font_family()``), falling back to Playfair
  Display, Libre Baskerville and Georgia.
- Body text and controls use Inter (``body_font_family()``), falling back to
  Helvetica Neue and Arial.

Hero and major section headings use the serif; controls and dense scientific
content use the sans-serif. Copy is plain and actionable.

Navigation
----------

The workflow reads as an expedition journal:

- The project navigator is the journal index. A completed stage shows a gold
  seal with a tick (``workflow_stage_icon``); an incomplete stage shows an
  empty dotted marker.
- The project home is a minimal chart drawn in code, with no bitmap artwork:
  the stages are waypoints on the plain page background, the navigator's
  sections are named regions,
  dependencies are inked, dashed or dotted routes, completed stages carry the
  same gold seal as the navigator, and a red flag marks where the user left
  off (see :doc:`/user_guide/project_home`).
