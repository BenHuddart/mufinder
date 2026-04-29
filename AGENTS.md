# Project Policy

- Delete-key functionality in Qt widgets must support both `QKeySequence.StandardKey.Delete` and `Qt.Key.Key_Backspace`. On macOS, the keyboard key labelled Delete is commonly delivered as Backspace, so any future delete/remove/cancel/reset shortcut should bind Backspace to the same handler as Delete and include a focused regression test.
