"""The imported marker must survive every round trip that touches a row.

Configuring any row re-registers the whole muon-relaxation definition from the
widget's current state, so a flag dropped anywhere in that loop silently unlocks
every imported row while its entities and files remain. These tests pin each leg
of the loop, and then the loop itself.

See ``docs/legacy_import.rst`` for why imported rows are read-only.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    serialize_muon_relaxation_config,
)


@pytest.fixture()
def main_window(qapp) -> MainWindow:
    """A bare main window; these tests touch only its row plumbing."""

    return MainWindow()


BATCH_ID = UUID("11111111-2222-3333-4444-555555555555")


def _imported_row(label: str = "Legacy Site 1", **overrides: object) -> MuonRelaxationRow:
    payload: dict[str, object] = {
        "site_id": uuid4(),
        "label": label,
        "initial_fractional_position": (0.1, 0.2, 0.3),
        "charge_state": "positive_muon",
        "imported": True,
        "import_batch_id": BATCH_ID,
    }
    payload.update(overrides)
    return MuonRelaxationRow(**payload)  # type: ignore[arg-type]


def _native_row(label: str = "Site 1") -> MuonRelaxationRow:
    return MuonRelaxationRow(
        site_id=uuid4(),
        label=label,
        initial_fractional_position=(0.4, 0.5, 0.6),
    )


def _widget_state(*rows: MuonRelaxationRow) -> dict[str, object]:
    """Build widget-shaped form state for ``rows``.

    The widget's state shape is **not** the definition's configuration shape:
    it names the position ``fractional_position`` where the configuration says
    ``initial_fractional_position``. Feeding one to the other silently zeroes
    every position, so these tests build the widget shape explicitly rather
    than reusing the configuration serialiser.
    """

    return {
        "rows": [
            {
                "site_id": str(row.site_id),
                "label": row.label,
                "fractional_position": list(row.initial_fractional_position),
                "charge_state": row.charge_state,
                "supercell_matrix": None,
                "parameter_values": dict(row.parameter_values),
                "added_advanced": list(row.added_advanced),
                "imported": row.imported,
                "import_batch_id": (
                    str(row.import_batch_id) if row.import_batch_id is not None else None
                ),
            }
            for row in rows
        ]
    }


class TestRowModel:
    def test_a_row_is_native_unless_marked(self) -> None:
        """Additive defaults mean a project saved before the import feature
        loads with every row treated as natively staged, which is correct."""

        row = _native_row()

        assert row.imported is False
        assert row.import_batch_id is None

    def test_an_imported_row_carries_its_batch(self) -> None:
        row = _imported_row()

        assert row.imported is True
        assert row.import_batch_id == BATCH_ID


class TestSerializerRoundTrip:
    def test_the_marker_survives_serialize_then_deserialize(self, main_window) -> None:
        config = MuonRelaxationConfig(rows=(_imported_row(), _native_row()))

        payload = serialize_muon_relaxation_config(config)
        restored = deserialize_muon_relaxation_config(payload)

        assert restored is not None
        assert [row.imported for row in restored.rows] == [True, False]
        assert restored.rows[0].import_batch_id == BATCH_ID
        assert restored.rows[1].import_batch_id is None

    def test_a_payload_predating_the_feature_deserializes_as_native(self, main_window) -> None:
        """Older project files carry no marker at all."""

        payload = serialize_muon_relaxation_config(MuonRelaxationConfig(rows=(_native_row(),)))
        for entry in payload["rows"]:
            entry.pop("imported", None)
            entry.pop("import_batch_id", None)

        restored = deserialize_muon_relaxation_config(payload)

        assert restored is not None
        assert restored.rows[0].imported is False

    def test_a_malformed_batch_id_does_not_unlock_the_row(self, main_window) -> None:
        """Losing the grouping is recoverable; losing read-only enforcement is not."""

        payload = serialize_muon_relaxation_config(MuonRelaxationConfig(rows=(_imported_row(),)))
        payload["rows"][0]["import_batch_id"] = "not-a-uuid"

        restored = deserialize_muon_relaxation_config(payload)

        assert restored is not None
        assert restored.rows[0].imported is True
        assert restored.rows[0].import_batch_id is None


class TestWidgetStateRoundTrip:
    def test_the_marker_survives_set_form_state_then_collect_state(self, main_window) -> None:
        widget = main_window._muon_relaxation_view
        imported, native = _imported_row(), _native_row()
        widget.set_form_state(_widget_state(imported, native))

        collected = widget.collect_state()

        by_label = {entry["label"]: entry for entry in collected["rows"]}
        assert by_label["Legacy Site 1"]["imported"] is True
        assert by_label["Legacy Site 1"]["import_batch_id"] == str(BATCH_ID)
        assert by_label["Site 1"]["imported"] is False
        assert by_label["Site 1"]["import_batch_id"] is None

    def test_a_malformed_batch_id_in_widget_state_keeps_the_row_imported(self, main_window) -> None:
        widget = main_window._muon_relaxation_view
        state = _widget_state(_imported_row())
        state["rows"][0]["import_batch_id"] = "not-a-uuid"

        widget.set_form_state(state)

        collected = widget.collect_state()
        assert collected["rows"][0]["imported"] is True
        assert collected["rows"][0]["import_batch_id"] is None

    def test_removing_a_row_clears_its_import_state(self, main_window) -> None:
        """Row ids are reused, so stale entries would mark the wrong row."""

        widget = main_window._muon_relaxation_view
        widget.set_form_state(_widget_state(_imported_row()))
        assert widget._row_imported

        widget.set_form_state({"rows": []})

        assert widget._row_imported == set()
        assert widget._row_import_batch_ids == {}


class TestReconfigureLoop:
    def test_the_marker_survives_a_full_widget_round_trip(self, main_window) -> None:
        """The path that drops it: configuring any row rebuilds the definition
        from the widget's current state, so the marker has to make the whole
        lap -- serialize, into the widget, back out, and into a row again."""

        widget = main_window._muon_relaxation_view
        imported, native = _imported_row(), _native_row()

        widget.set_form_state(_widget_state(imported, native))
        # The reconfigure path rebuilds the definition from the widget's
        # projected rows, so go back out through the config serialiser.
        projected = widget._build_config_or_warn()
        assert projected is not None
        round_tripped = deserialize_muon_relaxation_config(
            serialize_muon_relaxation_config(projected)
        )

        assert round_tripped is not None
        by_label = {row.label: row for row in round_tripped.rows}
        assert by_label["Legacy Site 1"].imported is True
        assert by_label["Legacy Site 1"].import_batch_id == BATCH_ID
        assert by_label["Site 1"].imported is False

    def test_the_projected_config_reports_the_marker(self, main_window) -> None:
        """The widget projects rows into a config for submission and staging."""

        widget = main_window._muon_relaxation_view
        widget.set_form_state(_widget_state(_imported_row(), _native_row()))

        config = widget._build_config_or_warn()

        assert config is not None
        by_label = {row.label: row for row in config.rows}
        assert by_label["Legacy Site 1"].imported is True
        assert by_label["Legacy Site 1"].import_batch_id == BATCH_ID
        assert by_label["Site 1"].imported is False


class TestRowSignature:
    def test_losing_the_marker_changes_the_signature(self, main_window) -> None:
        """Signature equality gates whether a row renders with its real status.

        Including the marker means a dropped flag makes the row visibly fall
        back to "ready" rather than rendering correctly while silently becoming
        editable.
        """

        imported = _imported_row()
        native = MuonRelaxationRow(
            site_id=imported.site_id,
            label=imported.label,
            initial_fractional_position=imported.initial_fractional_position,
            charge_state=imported.charge_state,
        )

        assert main_window._muon_relaxation_row_signature(
            imported
        ) != main_window._muon_relaxation_row_signature(native)

    def test_two_identical_imported_rows_share_a_signature(self, main_window) -> None:
        row = _imported_row()
        same = _imported_row(site_id=row.site_id, label=row.label)

        assert main_window._muon_relaxation_row_signature(
            row
        ) == main_window._muon_relaxation_row_signature(same)

    def test_rows_from_different_batches_do_not_share_a_signature(self, main_window) -> None:
        row = _imported_row()
        other = _imported_row(site_id=row.site_id, label=row.label, import_batch_id=uuid4())

        assert main_window._muon_relaxation_row_signature(
            row
        ) != main_window._muon_relaxation_row_signature(other)
