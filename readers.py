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




