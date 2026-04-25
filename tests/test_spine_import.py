from __future__ import annotations

import json
import zipfile

from flatrig_retarget_sidecar.spine_import import load_spine_package
from tests.helpers import build_spine_payload, write_spine_json


def test_load_spine_package_from_json(tmp_path) -> None:
    source = write_spine_json(tmp_path / "hero.json")

    package = load_spine_package(source)

    assert package.summary()["bone_count"] == 4
    assert package.bones_by_name["torso"].slot_count == 1
    assert package.animations["idle"]


def test_load_spine_package_from_zip_archive(tmp_path) -> None:
    archive_path = tmp_path / "hero.zip"
    payload = build_spine_payload()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/hero.json", json.dumps(payload))

    package = load_spine_package(f"{archive_path}!/nested/hero.json")

    assert package.summary()["animation_count"] == 1
    assert package.source_label.endswith("nested/hero.json")
