"""Tests for the missing-keyword-catalog indicator, prompt, and submit warning."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox, QWidget

import mufinder.app.keyword_catalog_guard as guard_module
import mufinder.castep_io.schema as schema_module
from mufinder.app.keyword_catalog_guard import (
    CHOICE_CANCEL,
    CHOICE_CONTINUE,
    CHOICE_SETUP,
    SKIP_STARTUP_PROMPT_KEY,
    KeywordCatalogGuard,
)
from mufinder.app.keyword_catalog_notice import (
    catalog_detail_text,
    catalog_short_label,
    catalog_source_description,
)
from mufinder.castep_io.schema import (
    CatalogSourceInfo,
    KeywordCatalogStatus,
    refresh_keyword_catalog,
)
from tests.support.keyword_catalogs import write_complete_catalog


class _FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 - Qt API naming.
        self.values[key] = value


def _use_catalog(monkeypatch: pytest.MonkeyPatch, path: Path | None, origin: str = "setup") -> None:
    sources = () if path is None else (CatalogSourceInfo(path=path, origin=origin),)
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", lambda: sources)
    refresh_keyword_catalog(allow_fallback=True)


class _Harness:
    """A guard whose modal prompt is answered from a scripted list."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: list[tuple[str, bool]]) -> None:
        self.parent = QWidget()
        self.settings = _FakeSettings()
        self.setup_opened = 0
        self.prompts: list[dict[str, object]] = []
        self.on_setup = lambda: None
        answers = list(answers)

        def show_choice(_guard, **kwargs):
            self.prompts.append(kwargs)
            return answers.pop(0)

        monkeypatch.setattr(KeywordCatalogGuard, "_show_choice", show_choice)
        self.guard = KeywordCatalogGuard(
            parent=self.parent, settings=self.settings, open_setup=self._open_setup
        )

    def _open_setup(self) -> None:
        self.setup_opened += 1
        self.on_setup()


def test_status_button_flags_the_limited_set_and_opens_setup(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _use_catalog(monkeypatch, None)
    harness = _Harness(monkeypatch, [])
    button = harness.guard.status_button

    assert button.text() == "⚠ CASTEP keywords: limited built-in set"
    assert "No keyword catalog is loaded" in button.toolTip()
    assert "Click to open CASTEP setup." in button.toolTip()
    assert button.property("catalogLimited") is True
    button.click()
    assert harness.setup_opened == 1

    _use_catalog(monkeypatch, write_complete_catalog(tmp_path / "castep_keywords.json", "26.11"))
    harness.guard.refresh()
    assert button.text() == "CASTEP keywords: 26.11"
    assert button.property("catalogLimited") is False
    assert "installed through CASTEP setup" in button.toolTip()


def test_status_button_flags_a_skipped_catalog_even_when_another_loaded(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    good = write_complete_catalog(tmp_path / "castep_keywords.json", "26.11")
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (
            CatalogSourceInfo(
                path=tmp_path / "gone.json", origin="env_json:MUFINDER_CASTEP_KEYWORDS_JSON"
            ),
            CatalogSourceInfo(path=good, origin="setup"),
        ),
    )
    refresh_keyword_catalog(allow_fallback=True)

    harness = _Harness(monkeypatch, [])

    assert harness.guard.status_button.text() == "⚠ CASTEP keywords: 26.11"
    assert "gone.json' no longer exists" in harness.guard.status_button.toolTip()


def test_startup_prompt_is_silent_with_a_catalog(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_catalog(monkeypatch, write_complete_catalog(tmp_path / "castep_keywords.json"))
    harness = _Harness(monkeypatch, [])

    harness.guard.prompt_at_startup_if_needed()

    assert harness.prompts == []


def test_startup_prompt_explains_and_opens_setup(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_catalog(monkeypatch, None)
    harness = _Harness(monkeypatch, [(CHOICE_SETUP, False)])

    harness.guard.prompt_at_startup_if_needed()

    prompt = harness.prompts[0]
    assert prompt["title"] == "CASTEP keyword catalog needed"
    assert "castep_keywords.json" in prompt["informative_text"]
    assert "QUANTISATION_AXIS" in prompt["informative_text"]
    assert prompt["offer_skip"] is True
    assert harness.setup_opened == 1
    assert SKIP_STARTUP_PROMPT_KEY not in harness.settings.values


def test_startup_prompt_can_be_dismissed_for_good(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_catalog(monkeypatch, None)
    harness = _Harness(monkeypatch, [(CHOICE_CONTINUE, True)])

    harness.guard.prompt_at_startup_if_needed()
    harness.guard.prompt_at_startup_if_needed()

    assert len(harness.prompts) == 1
    assert harness.setup_opened == 0
    assert harness.settings.values[SKIP_STARTUP_PROMPT_KEY] is True

    # QSettings hands booleans back as text on some platforms.
    harness.settings.values[SKIP_STARTUP_PROMPT_KEY] = "true"
    harness.guard.prompt_at_startup_if_needed()
    assert len(harness.prompts) == 1


def test_startup_prompt_says_why_a_configured_catalog_was_dropped(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "castep_keywords.json"
    corrupt.write_text("{", encoding="utf-8")
    _use_catalog(monkeypatch, corrupt)
    harness = _Harness(monkeypatch, [(CHOICE_CANCEL, False)])

    harness.guard.prompt_at_startup_if_needed()

    assert "could not be used" in harness.prompts[0]["informative_text"]
    assert harness.setup_opened == 0


def test_submission_is_not_questioned_with_a_catalog(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_catalog(monkeypatch, write_complete_catalog(tmp_path / "castep_keywords.json"))
    harness = _Harness(monkeypatch, [])

    assert harness.guard.confirm_submission() is True
    assert harness.prompts == []


def test_submission_warning_allows_submitting_anyway_once_per_session(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_catalog(monkeypatch, None)
    harness = _Harness(monkeypatch, [(CHOICE_CONTINUE, False)])

    assert harness.guard.confirm_submission() is True
    assert harness.guard.confirm_submission() is True

    assert len(harness.prompts) == 1
    assert harness.prompts[0]["continue_label"] == "Submit Anyway"
    assert harness.prompts[0]["offer_skip"] is False
    assert "may be unable to read the results" in harness.prompts[0]["informative_text"]


def test_submission_warning_cancel_blocks_and_asks_again(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_catalog(monkeypatch, None)
    harness = _Harness(monkeypatch, [(CHOICE_CANCEL, False), (CHOICE_CANCEL, False)])

    assert harness.guard.confirm_submission() is False
    assert harness.guard.confirm_submission() is False
    assert len(harness.prompts) == 2


def test_submission_proceeds_only_if_setup_installed_a_catalog(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _use_catalog(monkeypatch, None)
    harness = _Harness(monkeypatch, [(CHOICE_SETUP, False), (CHOICE_SETUP, False)])

    # The user opened setup but closed it without installing anything.
    assert harness.guard.confirm_submission() is False

    catalog = write_complete_catalog(tmp_path / "castep_keywords.json")
    harness.on_setup = lambda: _use_catalog(monkeypatch, catalog)
    assert harness.guard.confirm_submission() is True
    assert harness.setup_opened == 2


@pytest.mark.parametrize(
    ("button_text", "offer_skip", "tick_skip", "expected"),
    [
        ("Open CASTEP Setup…", True, False, (CHOICE_SETUP, False)),
        ("Not Now", True, True, (CHOICE_CONTINUE, True)),
        ("Submit Anyway", False, False, (CHOICE_CONTINUE, False)),
        ("Cancel", False, False, (CHOICE_CANCEL, False)),
    ],
)
def test_show_choice_maps_buttons_to_choices(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    button_text: str,
    offer_skip: bool,
    tick_skip: bool,
    expected: tuple[str, bool],
) -> None:
    monkeypatch.undo()  # use the real prompt rather than the conftest stand-in
    seen: dict[str, object] = {}

    def fake_exec(box: QMessageBox) -> int:
        seen["text"] = box.text()
        seen["buttons"] = [button.text().replace("&", "") for button in box.buttons()]
        if tick_skip:
            box.checkBox().setChecked(True)
        next(
            button for button in box.buttons() if button.text().replace("&", "") == button_text
        ).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    parent = QWidget()
    guard = KeywordCatalogGuard(parent=parent, settings=_FakeSettings(), open_setup=lambda: None)

    choice = guard._show_choice(
        title="No CASTEP keyword catalog",
        text="text",
        informative_text="more",
        continue_label="Not Now" if offer_skip else "Submit Anyway",
        offer_skip=offer_skip,
    )

    assert choice == expected
    # macOS drops message-box window titles, so the text carries the message.
    assert seen["text"] == "text"
    assert ("Cancel" in seen["buttons"]) is (not offer_skip)


def test_catalog_wording_for_each_origin() -> None:
    def _status(
        origin: str, *, version: str = "26.11", issue: str | None = None
    ) -> KeywordCatalogStatus:
        return KeywordCatalogStatus(
            origin=origin,
            path=Path("/data/castep_keywords.json"),
            is_starter=origin == "bundled_starter_catalog",
            castep_version=version,
            cell_count=118,
            param_count=449,
            issue=issue,
        )

    setup = _status("setup")
    assert catalog_short_label(setup) == "CASTEP keywords: 26.11"
    assert catalog_source_description(setup) == (
        "Keyword catalog for CASTEP 26.11 (118 cell and 449 param keywords), "
        "installed through CASTEP setup:\n/data/castep_keywords.json"
    )
    assert catalog_detail_text(setup) == catalog_source_description(setup)

    override = _status("env_json:MUFINDER_CASTEP_KEYWORDS_JSON", version="")
    assert catalog_short_label(override) == "CASTEP keywords: loaded"
    assert catalog_source_description(override) == (
        "Keyword catalog for an unrecorded CASTEP version (118 cell and 449 param keywords), "
        "set by the MUFINDER_CASTEP_KEYWORDS_JSON environment variable, which overrides "
        "CASTEP setup:\n/data/castep_keywords.json"
    )

    starter = _status(
        "bundled_starter_catalog", version="", issue="The configured keyword catalog is gone."
    )
    assert catalog_short_label(starter) == "CASTEP keywords: limited built-in set"
    detail = catalog_detail_text(starter)
    assert detail.index("No keyword catalog is loaded") < detail.index(
        "The configured keyword catalog is gone."
    )
    assert detail.endswith("or to import an existing castep_keywords.json.")


def test_guard_module_exports_choices() -> None:
    assert {
        guard_module.CHOICE_SETUP,
        guard_module.CHOICE_CONTINUE,
        guard_module.CHOICE_CANCEL,
    } == {
        "setup",
        "continue",
        "cancel",
    }
