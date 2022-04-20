from enum import IntEnum
from typing import NamedTuple


class DetectorBinning(IntEnum):
    BINNING1x1 = 1
    BINNING2x2 = 2
    BINNING4x4 = 4


class ArtisQSystem:
    def __init__(self, detector_binning: DetectorBinning):
        self.nb_pixels = (2480//detector_binning, 1920//detector_binning)
        self.pixel_dims = (0.154*detector_binning, 0.154*detector_binning)
        self.carm_span = 1200.0  # mm


class ConeGeometry(NamedTuple):
    det_count_u: int
    det_count_v: int
    det_spacing_u: float
    det_spacing_v: float
    src_dist: float
    det_dist: float
    pitch: float
    base_z: float


def default_cone_geometry() -> ConeGeometry:
    return ConeGeometry(
        det_count_u=620//2,  # detector columns in px
        det_count_v=480//2,  # detector rows in px
        det_spacing_u=.616*2,  # width of a detector pixel in mm
        det_spacing_v=.616*2,  # height of a detector pixel in mm
        src_dist=160,  # distance source to isocenter in mm
        det_dist=240,  # distance detector to isocenter in mm
        pitch=0.0,  # pitch factor (not needed for circular trajectory)
        base_z=0.0,  # reference z position wrt. the volume
    )
