import math
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET

# load aint data
def load_aint(tracks_ffn):
    """
    Function to load AINT track data from a CSV file.

    Parameters
    ----------
    tracks_ffn : str
        Path to the AINT track CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the AINT track data with columns:
        track_id, timestamp, lat, lon, area, initial_track_id, filter
    """
    df = pd.read_csv(tracks_ffn, parse_dates=['time'])
    #rename
    df = df.rename(columns={'time': 'timestamp', 'uid': 'track_id'})
    #copy track_id to initial_track_id for later reference
    df['initial_track_id'] = df['track_id']
    #add filter column
    df['filter'] = None
    return df

def load_aint_national(tracks_ffn: str) -> pd.DataFrame:
    """
    Load a merged AINT national tracks CSV file.

    The file contains '#'-prefixed metadata/comment lines followed by a
    standard CSV with columns including uid, time, lat, lon, area_km2.

    Parameters
    ----------
    tracks_ffn : str
        Path to the merged AINT national tracks CSV file
        (e.g. ``310_20251101_merged.tracks.csv``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: track_id, timestamp, lat, lon, area,
        initial_track_id, filter.
    """
    df = pd.read_csv(tracks_ffn, delimiter=',', comment='#', parse_dates=['time'])
    df = df.rename(columns={'uid': 'track_id', 'time': 'timestamp', 'area_km2': 'area'})
    df['initial_track_id'] = df['track_id']
    df['filter'] = None
    return df[['track_id', 'timestamp', 'lat', 'lon', 'area', 'initial_track_id', 'filter']]


def load_titan_ascii(ascii_ffn: str) -> pd.DataFrame:
    """
    Read a storms_to_tifs ASCII file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the storms_to_tifs file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the TITAN track data with columns:
        track_id, timestamp, lat, lon, area, initial_track_id, filter
    """
    with open(ascii_ffn, "r") as f:
        all_lines = f.readlines()

    # Row 20 is the comma-delimited header; fix duplicate 'mm' column
    header_line = all_lines[19]
    columns = [c.strip() for c in header_line.split(",")]
    seen = {}
    for i, col in enumerate(columns):
        if col in seen:
            columns[i] = "min" if col == "mm" else f"{col}_{seen[col]}"
            seen[col] += 1
        else:
            seen[col] = 1

    # Data starts at row 21, whitespace-delimited
    df = pd.read_csv(
        StringIO("".join(all_lines[20:])),
        header=None,
        sep=r"\s+",
        names=columns,
    )

    # Build datetime from individual time columns
    df["timestamp"] = pd.to_datetime(
        df[["yyyy", "mm", "dd", "hh", "min", "ss"]].rename(
            columns={"yyyy": "year", "mm": "month", "dd": "day",
                      "hh": "hour", "min": "minute", "ss": "second"}
        )
    )

    # Select and rename columns
    df = df.rename(columns={
        "simple_tk": "track_id",
        "lat(deg)": "lat",
        "long(deg)": "lon",
        "precip_area(km2)": "area",
    })[["track_id", "timestamp", "lat", "lon", "area"]]

    #add filter column
    df['filter'] = None
    #copy track_id to initial_track_id for later reference
    df['initial_track_id'] = df['track_id']
    return df

# load titan data
def load_titan_xml(xml_ffn_list):
    """
    Function to load TITAN track data from a list of XML files.

    Parameters
    ----------
    xml_ffn_list : list of str
        List of paths to the TITAN track XML files.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the TITAN track data with columns:
        track_id, timestamp, lat, lon, area, initial_track_id, filter
    """

    #init schema
    NS = {"w": "https://reg.bom.gov.au/schema/WxML"}
    rows = []
    for xml_ffn in xml_ffn_list:
        tree = ET.parse(xml_ffn)
        root = tree.getroot()
        for event in root.findall(".//w:event", NS):
            track_id = event.get("ID")
            for case in event.findall("w:case", NS):
                #skip if case_type is "historical" or "nowcast"
                if case.get("description") in ["history", "forecast"]:
                    continue
                #init a row using the track id
                row = {"track_id": track_id}
                # add the timestamp
                time_el = case.find("w:time", NS)
                row["timestamp"] = time_el.text if time_el is not None else None

                # Centroid from ellipse moving-point
                mp = case.find(".//w:ellipse/w:moving-point", NS)
                if mp is not None:
                    row["lat"] = float(mp.find("w:latitude", NS).text)
                    row["lon"] = float(mp.find("w:longitude", NS).text)
                else:
                    continue  # skip


                # Nowcast parameters
                area = case.find("w:nowcast-parameters/w:projected_area", NS)
                if area is not None:
                    row["area"] = float(area.text)
                else:
                    continue  # skip

                rows.append(row)

    df = pd.DataFrame(rows)

    #parse time column as datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    #copy track_id to initial_track_id for later reference
    df['initial_track_id'] = df['track_id']
    #add filter column
    df['filter'] = None
    return df


def read_ww_hailtracker(track_ffn) -> str:
    """
    Assign a unique integer track_id to each thunderstorm track, splitting
    at any point where multiple children share the same parent so that each
    track has at most one cell per timestep. The child geographically closest
    to the parent inherits the parent's track.

    Parameters
    ----------
    track_ffn : str
        ascii file with track data. Must contain columns 'storm_index' and 'storm_index_previous'.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with a new 'track_id' column inserted as the
        first column. Rows are sorted by track_id, then timestamp, then
        storm_index.
    """
    def _haversine_km(lon1, lat1, lon2, lat2):
        """Great-circle distance in km between two lon/lat points."""
        R = 6371.0
        dlon = math.radians(lon2 - lon1)
        dlat = math.radians(lat2 - lat1)
        a = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    df = pd.read_csv(track_ffn)
    df = df.sort_values(["timestamp", "storm_index"]).reset_index(drop=True)

    # Build lookups for cell locations
    loc = dict(zip(
        df["storm_index"],
        zip(df["centre_lon"], df["centre_lat"])
    ))

    # Track ID assigned to each storm_index
    track_for_storm = {}
    next_track_id = 1

    # Process cells in chronological order
    for timestamp, group in df.groupby("timestamp", sort=True):
        children_of = {}  # parent_storm_index -> list of child storm_indices
        roots = []        # cells with no parent (start new tracks)

        for _, row in group.iterrows():
            si = row["storm_index"]
            si_prev = row["storm_index_previous"]

            if pd.isna(si_prev):
                roots.append(si)
            else:
                parent = int(si_prev)
                children_of.setdefault(parent, []).append(si)

        # Cells with no parent always start a new track
        for si in roots:
            track_for_storm[si] = next_track_id
            next_track_id += 1

        # For each parent, assign tracks to its children
        for parent, children in children_of.items():
            # Sort children by distance to parent (closest first)
            parent_lon, parent_lat = loc[parent]
            children.sort(key=lambda si: _haversine_km(
                parent_lon, parent_lat, loc[si][0], loc[si][1]
            ))

            if parent in track_for_storm:
                parent_track = track_for_storm[parent]
            else:
                # Parent not seen (edge case) - treat first child as new track
                parent_track = next_track_id
                next_track_id += 1

            # Closest child continues the parent's track
            track_for_storm[children[0]] = parent_track

            # Remaining children each start a new track
            for si in children[1:]:
                track_for_storm[si] = next_track_id
                next_track_id += 1

    # Map track IDs onto the dataframe
    df["track_id"] = df["storm_index"].map(track_for_storm)

    # Renumber track_ids sequentially (1, 2, 3, ...) in order of first appearance
    first_appearance = df.sort_values(["timestamp", "storm_index"]).drop_duplicates("track_id")
    tid_remap = {old: new for new, old in enumerate(first_appearance["track_id"], start=1)}
    df["track_id"] = df["track_id"].map(tid_remap)

    # Reorder columns: track_id first
    cols = ["track_id"] + [c for c in df.columns if c != "track_id"]
    df = df[cols]

    # Sort for readability
    sort_cols = ["track_id"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    sort_cols.append("storm_index")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    #parse time column as datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Select and rename columns
    df = df.rename(columns={
        "centre_lat": "lat",
        "centre_lon": "lon",
    })[["track_id", "timestamp", "lat", "lon"]]

    #insert dummy area column
    df['area'] = 1
    
    return df



