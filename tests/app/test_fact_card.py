"""The fact card: groups of captioned values on one line when they fit."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from mufinder.app.fact_card import FactCard
from mufinder.app.theme import form_spacing

GROUPS = (
    (("Formula", "Co2F4"), ("Atoms", "6"), ("Space group", "P4₂/mnm (136)")),
    (("a", "4.6950 Å"), ("b", "4.6950 Å"), ("c", "3.1800 Å"), ("Volume", "70.10 Å³")),
)


def _shown(qapp, width: int, groups=GROUPS) -> tuple[QWidget, FactCard]:
    page = QWidget()
    layout = QVBoxLayout(page)
    card = FactCard(accessible_name="Facts")
    card.set_groups(groups)
    layout.addWidget(card)
    layout.addStretch(1)
    page.resize(width, 400)
    page.show()
    qapp.processEvents()
    return page, card


def _top(card: FactCard, caption: str) -> int:
    label = card.value_label(caption)
    return label.mapTo(card, label.rect().topLeft()).y()


def _left(card: FactCard, caption: str) -> int:
    label = card.value_label(caption)
    return label.mapTo(card, label.rect().topLeft()).x()


def _right(card: FactCard, caption: str) -> int:
    label = card.value_label(caption)
    return label.mapTo(card, label.rect().topRight()).x()


def test_groups_share_one_line_with_a_hairline_between_them(qapp) -> None:
    page, card = _shown(qapp, 1200)

    assert card.is_one_line()
    assert len({_top(card, caption) for group in GROUPS for caption, _ in group}) == 1
    (hairline,) = card.hairlines()
    assert hairline.isVisible()
    hairline_x = hairline.mapTo(card, hairline.rect().topLeft()).x()
    # Equal space either side of the hairline.
    left_gap = hairline_x - _right(card, "Space group") - 1
    right_gap = _left(card, "a") - hairline_x - 1
    assert abs(left_gap - right_gap) <= 1
    assert card.facts() == [list(group) for group in GROUPS]
    page.close()


def test_spare_width_widens_the_gaps_evenly_up_to_twice_their_size(qapp) -> None:
    page, card = _shown(qapp, 2400)

    gap = _left(card, "Atoms") - _right(card, "Formula") - 1
    assert gap == _left(card, "b") - _right(card, "a") - 1
    assert form_spacing() * 2 < gap <= form_spacing() * 4
    # The rest is left at the right end, not spread and not centred.
    assert _left(card, "Formula") < card.width() // 10
    assert _right(card, "Volume") < card.width() * 3 // 4
    page.close()


def test_a_narrow_card_breaks_at_the_hairline_then_fact_by_fact(qapp) -> None:
    # Narrower than the one line of facts needs, whatever the theme's fonts.
    page, card = _shown(qapp, 2000)
    page.resize(card.sizeHint().width() - 40, 400)
    qapp.processEvents()

    assert not card.is_one_line()
    assert not card.hairlines()[0].isVisible()
    assert _top(card, "a") > _top(card, "Formula")
    assert _top(card, "Volume") >= _top(card, "a")
    narrow_page, narrow = _shown(qapp, 180)
    assert _top(narrow, "Volume") > _top(narrow, "a")
    for caption in ("Space group", "Volume"):
        assert _right(narrow, caption) < narrow.width()
    page.close()
    narrow_page.close()


def test_a_long_value_is_shortened_before_the_line_breaks(qapp) -> None:
    """Elide first, down to the minimum, then break: a long path does not force two rows."""
    path = "imports/" + "a_very_long_structure_file_name_" * 6 + "end.cif"
    groups = (GROUPS[0] + (("Source", path),), GROUPS[1])
    page, card = _shown(qapp, 1200, groups)
    value = card.value_label("Source")

    assert card.is_one_line()
    assert value.text().startswith("imports/")
    assert value.text().endswith("end.cif")
    assert "…" in value.text()
    assert value.value() == path
    assert value.toolTip() == path
    assert value.accessibleDescription() == f"Source, {path}"
    assert card.minimumSizeHint().width() < 300
    page.close()


def test_each_value_is_selectable_and_tells_a_screen_reader_its_caption(qapp) -> None:
    page, card = _shown(qapp, 900)
    value = card.value_label("a")

    assert value.accessibleDescription() == "a, 4.6950 Å"
    assert value.accessibleName() == "a"
    assert card.accessibleName() == "Facts"
    page.close()


def test_one_group_has_no_hairline(qapp) -> None:
    page, card = _shown(qapp, 900, GROUPS[:1])

    assert card.hairlines() == []
    assert card.is_one_line()
    page.close()


def test_a_message_replaces_the_facts_and_facts_replace_the_message(qapp) -> None:
    card = FactCard(accessible_name="Facts")
    card.set_groups(GROUPS)
    card.set_message("Nothing to show.")

    assert card.facts() == []
    assert card.message() == "Nothing to show."
    assert card.value_label("a") is None
    assert card.property("factCardState") == "message"

    card.set_groups(GROUPS[:1])
    assert card.message() == ""
    assert card.facts() == [list(GROUPS[0])]
    assert card.property("factCardState") == "facts"


def test_copy_gives_the_whole_value_of_a_shortened_one(qapp) -> None:
    path = "imports/" + "a_very_long_structure_file_name_" * 6 + "end.cif"
    page, card = _shown(qapp, 700, ((("Source", path),),))
    value = card.value_label("Source")
    assert "…" in value.text()

    value.copy_value()

    assert QApplication.clipboard().text() == path
    page.close()
