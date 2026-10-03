"""Immutable source revisions and count checks for the released inputs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSnapshots:
    """Repository identities and immutable Git revisions used by one run."""

    labels_repo: str
    labels_revision: str
    eunis_repo: str
    eunis_revision: str
    input_revision: str


INPUT_REVISION = "b4706eb315c66a8a57135289e7612dca8e03caf8"
LANDUSE_LABELS_REPO = "NoeFlandre/osm-polygon-description-tag-landuse"
LANDUSE_LABELS_REVISION = "d52781febf61a1ee77e4aceb02d9b607628b37c5"
EUNIS_REPO = "NoeFlandre/osm-polygon-description-tag-eunis"
EUNIS_REVISION = "88eabd83fcc60cb67ab4c1768a16e3d05e25ffb3"
EXPECTED_SHARDS = 386

DEFAULT_SNAPSHOTS = DatasetSnapshots(
    labels_repo=LANDUSE_LABELS_REPO,
    labels_revision=LANDUSE_LABELS_REVISION,
    eunis_repo=EUNIS_REPO,
    eunis_revision=EUNIS_REVISION,
    input_revision=INPUT_REVISION,
)

EXPECTED_SOURCE_COUNTS = {
    "polygon_rows": 906_631,
    "assigned_polygon_rows": 276_604,
    "description_tag_rows": 919_126,
    "label_rows": 1_057_002,
    "sentence_rows": 978_773,
    "yes": 725_371,
    "no": 241_475,
    "failed": 11_927,
    "skipped_unsplit": 78_229,
}
