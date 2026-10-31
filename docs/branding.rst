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

Scales
------

``mufinder.app.theme`` holds the scales. Every font size, padding and corner
radius in its stylesheets comes from them:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Scale
     - Steps (px)
     - Used for
   * - ``TYPE_SCALE``
     - ``caption`` 11, ``body`` 13, ``emphasis`` 15, ``section`` 18, ``page`` 28
     - Font sizes. The application font, and so every control and table, is
       ``caption``.
   * - ``SPACING``
     - ``xs`` 4, ``sm`` 8, ``md`` 12, ``lg`` 16, ``xl`` 24
     - Padding, and the margins and gaps of layouts built with the helpers below
   * - ``RADIUS``
     - ``control`` 6, ``card`` 10
     - Corners of controls, tables and tabs; corners of cards and panels

A square corner or an absent padding is ``0px``. A round shape takes half its
side (``ROUND_RADIUS``), and a combo box or spin box keeps a gutter for its
arrow buttons, so neither is a step on a scale.

Text follows the desktop: ``type_scale()`` multiplies ``TYPE_SCALE`` by
``text_scale()``, the desktop's general font size over the usual default of
that platform (13 pt on macOS, 9 pt on Windows, 11 pt elsewhere), from 1 up to 2.
A default or smaller desktop font gives the sizes above, so the smallest step is
never under 11 px. The scale is fixed when a theme is applied, so a change of
desktop font takes effect at the next start or change of appearance. Painted
text and a page's own stylesheet keep their pixel sizes until the page is rebuilt.
A width cap on a control that holds text goes through ``scaled_to_text()``.

Layouts take ``page_margins()``, ``card_margins()`` and ``form_spacing()``, or a
``SPACING`` step. A page moves to these when it is next rebuilt; the harness
counts the literal sizes still passed outside ``theme.py``, and that count may
only fall.

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
