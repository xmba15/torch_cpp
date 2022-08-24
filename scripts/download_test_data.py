#!/usr/bin/env python
import os

from google_drive_downloader import GoogleDriveDownloader as gdd

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_DATA_DIR = os.path.realpath(os.path.join(_CURRENT_DIR, "../data"))


def main():
    # traced superpoint weights
    gdd.download_file_from_google_drive(
        file_id="1EjV07y940z46HscSFadPkHWlhAg1T5nc",
        dest_path=os.path.join(_DATA_DIR, "superpoint_model.pt"),
        unzip=True,
    )


if __name__ == "__main__":
    main()
