from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pyxtf.xtf_io import xtf_read
from pyxtf.enumerations import XTFHeaderType, XTFChannelType


@dataclass
class PingNavigation:
    ping_number: int
    event_number: int

    latitude: float
    longitude: float

    depth_m: float
    altitude_m: float

    heading_deg: float
    pitch_deg: float
    roll_deg: float
    heave_m: float


@dataclass
class SonarFrame:
    event_number: int

    image: np.ndarray

    pings: list[PingNavigation]

    channel_number: int
    channel_type: str

    ground_range_m: float
    slant_range_m: float

    width: int
    height: int

    source_file: str


@dataclass
class GeotaggedDetection:
    event_number: int
    ping_number: int

    class_id: int
    class_name: str
    confidence: float

    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    bbox_width: float
    bbox_height: float

    center_x_px: float
    center_y_px: float

    latitude: float
    longitude: float

    sensor_depth_m: float
    altitude_m: float
    heading_deg: float

    across_track_offset_m: float
    ground_range_from_sensor_m: float
    slant_range_from_sensor_m: float

    channel: str

    geotag_method: str = "XTF ping navigation + sonar pixel geometry"

    def to_dict(self):
        return asdict(self)


class XTFMissionReader:
    """
    Reads an XTF side-scan sonar mission and reconstructs
    event-level sonar frames.

    Each EventNumber is treated as one reconstructed sonar frame.
    Each ping contributes one image row.
    """

    def __init__(self, xtf_path: str | Path):
        self.path = Path(xtf_path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"XTF mission not found: {self.path}"
            )

        self.file_header, packets = xtf_read(
            str(self.path)
        )

        self.sonar_packets = packets.get(
            XTFHeaderType.sonar,
            []
        )

        if not self.sonar_packets:
            raise RuntimeError(
                "No sonar packets were found in the XTF mission."
            )

    def event_numbers(self):
        return sorted(
            {
                int(p.EventNumber)
                for p in self.sonar_packets
            }
        )

    def _get_channel_type(self, ping, channel_index):
        try:
            channel_number = (
                ping.ping_chan_headers[channel_index]
                .ChannelNumber
            )

            chan_info = self.file_header.ChanInfo[
                channel_number
            ]

            channel_type = chan_info.TypeOfChannel

            if channel_type == XTFChannelType.port.value:
                return "port"

            if channel_type == XTFChannelType.stbd.value:
                return "starboard"

        except Exception:
            pass

        return "unknown"

    def _navigation_from_ping(
        self,
        ping,
    ) -> PingNavigation:

        return PingNavigation(
            ping_number=int(ping.PingNumber),
            event_number=int(ping.EventNumber),

            latitude=float(ping.SensorYcoordinate),
            longitude=float(ping.SensorXcoordinate),

            depth_m=float(ping.SensorDepth),
            altitude_m=float(
                ping.SensorPrimaryAltitude
            ),

            heading_deg=float(
                ping.SensorHeading
            ),

            pitch_deg=float(
                ping.SensorPitch
            ),

            roll_deg=float(
                ping.SensorRoll
            ),

            heave_m=float(
                ping.Heave
            ),
        )

    def read_frames(self):
        """
        Yield one SonarFrame per EventNumber.
        """

        grouped = {}

        for ping in self.sonar_packets:
            grouped.setdefault(
                int(ping.EventNumber),
                []
            ).append(ping)

        for event_number in sorted(grouped):

            pings = grouped[event_number]

            pings.sort(
                key=lambda p: int(p.PingNumber)
            )

            rows = []
            navigation = []

            ground_range = 0.0
            slant_range = 0.0
            channel_type = "unknown"
            channel_number = 0

            for ping in pings:

                if not ping.data:
                    continue

                data = np.asarray(
                    ping.data[0],
                    dtype=np.uint8
                )

                rows.append(data)

                navigation.append(
                    self._navigation_from_ping(
                        ping
                    )
                )

                if ping.ping_chan_headers:

                    channel = (
                        ping.ping_chan_headers[0]
                    )

                    channel_number = int(
                        channel.ChannelNumber
                    )

                    ground_range = float(
                        channel.GroundRange
                    )

                    slant_range = float(
                        channel.SlantRange
                    )

                    channel_type = (
                        self._get_channel_type(
                            ping,
                            0
                        )
                    )

            if not rows:
                continue

            widths = {
                len(row)
                for row in rows
            }

            if len(widths) != 1:
                raise RuntimeError(
                    f"Event {event_number} has "
                    f"inconsistent sonar widths: "
                    f"{widths}"
                )

            image = np.vstack(rows)

            yield SonarFrame(
                event_number=event_number,

                image=image,

                pings=navigation,

                channel_number=channel_number,
                channel_type=channel_type,

                ground_range_m=ground_range,
                slant_range_m=slant_range,

                width=image.shape[1],
                height=image.shape[0],

                source_file=str(self.path),
            )

    def get_frame(self, event_number: int):
        for frame in self.read_frames():
            if frame.event_number == event_number:
                return frame

        raise KeyError(
            f"Event {event_number} not found."
        )

    def full_track(self):
        """
        Every ping's navigation fix, in file order - the ship's complete
        surveyed path for this mission, independent of which events were
        actually reconstructed into frames.

        Returns a list of PingNavigation. track[0] is the start of the
        journey, track[-1] is the end.
        """
        return [
            self._navigation_from_ping(ping)
            for ping in self.sonar_packets
            if ping is not None
        ]

    def journey_bounds(self):
        """
        (start, end) PingNavigation for the mission - the first and last
        navigation fix recorded in the XTF file, in the order pings were
        logged (not sorted by event/ping number, so this reflects the
        actual start and end of the vessel's run).
        """
        track = self.full_track()
        if not track:
            raise RuntimeError("No navigation data found in this XTF mission.")
        return track[0], track[-1]


def _interpolate_ping(
    frame: SonarFrame,
    y_pixel: float,
):
    """
    Convert image Y coordinate into the corresponding
    sonar ping navigation record.

    The image row is directly associated with a ping,
    so this is a 1D interpolation along the survey track.
    """

    if not frame.pings:
        raise RuntimeError(
            "Frame contains no navigation records."
        )

    y = float(
        np.clip(
            y_pixel,
            0,
            frame.height - 1
        )
    )

    lower = int(np.floor(y))
    upper = min(
        lower + 1,
        len(frame.pings) - 1
    )

    alpha = y - lower

    p1 = frame.pings[lower]
    p2 = frame.pings[upper]

    latitude = (
        p1.latitude * (1.0 - alpha)
        + p2.latitude * alpha
    )

    longitude = (
        p1.longitude * (1.0 - alpha)
        + p2.longitude * alpha
    )

    depth = (
        p1.depth_m * (1.0 - alpha)
        + p2.depth_m * alpha
    )

    altitude = (
        p1.altitude_m * (1.0 - alpha)
        + p2.altitude_m * alpha
    )

    h1 = float(p1.heading_deg) % 360.0
    h2 = float(p2.heading_deg) % 360.0

    delta = ((h2 - h1 + 180.0) % 360.0) - 180.0

    heading = (h1 + alpha * delta) % 360.0

    return (
        lower,
        latitude,
        longitude,
        depth,
        altitude,
        heading,
    )


def _offset_lat_lon(
    latitude,
    longitude,
    offset_m,
    bearing_deg,
):
    """
    Small-distance local ENU approximation.

    Suitable for the short spatial extent of an individual
    sonar frame / demo mission.
    """

    earth_radius = 6378137.0

    bearing = np.deg2rad(
        bearing_deg
    )

    north_m = (
        offset_m *
        np.cos(bearing)
    )

    east_m = (
        offset_m *
        np.sin(bearing)
    )

    lat_rad = np.deg2rad(latitude)

    d_lat = (
        north_m /
        earth_radius
    )

    d_lon = (
        east_m /
        (
            earth_radius *
            np.cos(lat_rad)
        )
    )

    return (
        latitude
        + np.rad2deg(d_lat),

        longitude
        + np.rad2deg(d_lon),
    )


def geotag_detection(
    frame: SonarFrame,
    detection: dict,
):
    """
    Convert a YOLO detection in pixel coordinates
    into a georeferenced detection.

    Assumption:
      image Y = ping / along-track direction
      image X = across-track direction
    """

    x1, y1, x2, y2 = (
        float(detection["bbox"][0]),
        float(detection["bbox"][1]),
        float(detection["bbox"][2]),
        float(detection["bbox"][3]),
    )

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    (
        ping_index,
        latitude,
        longitude,
        depth,
        altitude,
        heading,
    ) = _interpolate_ping(
        frame,
        cy
    )

    # Pixel position relative to the centerline.
    normalized_x = (
        cx / max(frame.width - 1, 1)
    )

    normalized_x = (
        normalized_x - 0.5
    ) * 2.0

    # Treat GroundRange as the maximum
    # across-track range represented by the frame.
    across_track = (
        normalized_x *
        frame.ground_range_m
    )

    # Port = left of vehicle heading.
    # Starboard = right of vehicle heading.
    if frame.channel_type == "port":
        bearing = heading - 90.0

    elif frame.channel_type == "starboard":
        bearing = heading + 90.0

    else:
        bearing = heading + 90.0

    ground_range_from_sensor = abs(across_track)

    # First-order 3D slant range.
    # Sensor altitude is the vertical component and the
    # across-track ground range is the horizontal component.
    slant_range_from_sensor = float(
        np.sqrt(
            ground_range_from_sensor ** 2 +
            max(float(altitude), 0.0) ** 2
        )
    )

    latitude, longitude = _offset_lat_lon(
        latitude,
        longitude,
        across_track,
        bearing
    )

    return GeotaggedDetection(
        event_number=frame.event_number,
        ping_number=frame.pings[ping_index].ping_number,

        class_id=int(
            detection["class_id"]
        ),

        class_name=str(
            detection["class_name"]
        ),

        confidence=float(
            detection["confidence"]
        ),

        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,

        bbox_width=x2 - x1,
        bbox_height=y2 - y1,

        center_x_px=cx,
        center_y_px=cy,

        latitude=latitude,
        longitude=longitude,

        sensor_depth_m=depth,
        altitude_m=altitude,
        heading_deg=heading,

        across_track_offset_m=across_track,

        ground_range_from_sensor_m=ground_range_from_sensor,

        slant_range_from_sensor_m=slant_range_from_sensor,

        channel=frame.channel_type,
    )