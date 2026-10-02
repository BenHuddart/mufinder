"""Keeps the user aware of a missing CASTEP keyword catalog.

Running without a ``castep_keywords.json`` is allowed but degraded, and nothing
about the degraded state is obvious from the workflow tabs. The guard makes it
visible in three places: a permanent status-bar indicator, a one-off prompt when
the application starts, and a warning before jobs are submitted. All three lead
to the CASTEP setup dialog; none of them blocks the user from continuing.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QSettings
from PySide6.QtWidgets import QCheckBox, QMessageBox, QPushButton, QWidget

from mufinder.app.keyword_catalog_notice import (
    LIMITED_CATALOG_EXPLANATION,
    catalog_detail_text,
    catalog_short_label,
)
from mufinder.castep_io.schema import keyword_catalog_status


SKIP_STARTUP_PROMPT_KEY = "castep/skip_keyword_catalog_prompt"

#: Choices returned by :meth:`KeywordCatalogGuard._show_choice`.
CHOICE_SETUP = "setup"
CHOICE_CONTINUE = "continue"
CHOICE_CANCEL = "cancel"


class KeywordCatalogGuard(QObject):
    """Surface the keyword-catalog state and route the user to CASTEP setup."""

    def __init__(
        self,
        *,
        parent: QWidget,
        settings: QSettings,
        open_setup: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._settings = settings
        self._open_setup = open_setup
        # Once the user has chosen to submit on the limited set, do not ask
        # again for every later submission in the same session.
        self._submission_acknowledged = False

        self.status_button = QPushButton(parent)
        self.status_button.setFlat(True)
        self.status_button.setObjectName("keywordCatalogStatusButton")
        self.status_button.clicked.connect(lambda *_: self._open_setup())
        self.refresh()

    def refresh(self) -> None:
        """Re-read the catalog state into the status-bar indicator."""

        status = keyword_catalog_status()
        label = catalog_short_label(status)
        needs_attention = status.is_starter or bool(status.issue)
        self.status_button.setText(f"⚠ {label}" if needs_attention else label)
        self.status_button.setToolTip(f"{catalog_detail_text(status)}\n\nClick to open CASTEP setup.")
        self.status_button.setProperty("catalogLimited", needs_attention)

    def prompt_at_startup_if_needed(self) -> None:
        """Explain a missing catalog once per launch, unless the user opted out."""

        status = keyword_catalog_status()
        if not status.is_starter:
            return
        if _as_bool(self._settings.value(SKIP_STARTUP_PROMPT_KEY, False)):
            return

        text = "MuFinder is running without a CASTEP keyword catalog."
        details = LIMITED_CATALOG_EXPLANATION
        if status.issue:
            details = f"{status.issue}\n\n{details}"
        choice, skip_in_future = self._show_choice(
            title="CASTEP keyword catalog needed",
            text=text,
            informative_text=f"{details}\n\nIt takes a minute or two to set up.",
            continue_label="Not Now",
            offer_skip=True,
        )
        if skip_in_future:
            self._settings.setValue(SKIP_STARTUP_PROMPT_KEY, True)
        if choice == CHOICE_SETUP:
            self._open_setup()

    def confirm_submission(self) -> bool:
        """Return whether a submission should go ahead on the current catalog.

        With a full catalog this is always true and shows nothing. On the
        limited built-in set the user is warned and may submit anyway, set the
        catalog up first, or cancel.
        """

        if self._submission_acknowledged or not keyword_catalog_status().is_starter:
            return True

        choice, _skip = self._show_choice(
            title="No CASTEP keyword catalog",
            text="These jobs will be submitted without a CASTEP keyword catalog.",
            informative_text=(
                "MuFinder may be unable to read the results if CASTEP writes a keyword that its limited "
                "built-in set does not list. Setting the catalog up first takes a minute or two."
            ),
            continue_label="Submit Anyway",
            offer_skip=False,
        )
        if choice == CHOICE_SETUP:
            self._open_setup()
            # Setting the catalog up resolves the warning; otherwise the user
            # is asked again on their next attempt.
            return not keyword_catalog_status().is_starter
        if choice == CHOICE_CONTINUE:
            self._submission_acknowledged = True
            return True
        return False

    def _show_choice(
        self,
        *,
        title: str,
        text: str,
        informative_text: str,
        continue_label: str,
        offer_skip: bool,
    ) -> tuple[str, bool]:
        """Show the modal prompt and return ``(choice, skip_in_future)``."""

        box = QMessageBox(self._parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText(informative_text)
        setup_button = box.addButton("Open CASTEP Setup…", QMessageBox.ButtonRole.AcceptRole)
        continue_button = box.addButton(continue_label, QMessageBox.ButtonRole.DestructiveRole)
        skip_box: QCheckBox | None = None
        if offer_skip:
            skip_box = QCheckBox("Don't show this again", box)
            box.setCheckBox(skip_box)
            box.setEscapeButton(continue_button)
        else:
            box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(setup_button)
        box.exec()

        clicked = box.clickedButton()
        skip_in_future = bool(skip_box is not None and skip_box.isChecked())
        if clicked is setup_button:
            return CHOICE_SETUP, skip_in_future
        if clicked is continue_button:
            return CHOICE_CONTINUE, skip_in_future
        return CHOICE_CANCEL, skip_in_future


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
