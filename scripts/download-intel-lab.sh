#!/usr/bin/env bash
# Download the Intel Research Lab dataset (CARMEN log format).
# Source: Dirk Haehnel, Intel Research Seattle, 2003.
set -euo pipefail

DATA_DIR="data/intel-lab"
mkdir -p "$DATA_DIR"

# The dataset is available from multiple mirrors.
# Primary: Radish (Robotics Data Set Repository)
URL="https://raw.githubusercontent.com/1988kramer/intel_dataset/master/data/intel.gfs.log"

# Alternative raw Carmen log (if GFS-processed version not suitable):
# The GFS log contains FLASER entries which are what we need.
URL_RAW="https://raw.githubusercontent.com/xiaofeng419/SLAM-2D-LIDAR-SCAN/master/DataSet/RawData/intel.gfs"

echo "Downloading Intel Research Lab dataset..."

if [ -f "$DATA_DIR/intel.gfs" ]; then
    echo "Dataset already exists at $DATA_DIR/intel.gfs"
    echo "Delete it to re-download."
    exit 0
fi

# Try primary source first
if curl -fsSL -o "$DATA_DIR/intel.gfs" "$URL_RAW" 2>/dev/null; then
    echo "Downloaded from GitHub (xiaofeng419)"
elif curl -fsSL -o "$DATA_DIR/intel.gfs" "$URL" 2>/dev/null; then
    echo "Downloaded from GitHub (1988kramer)"
else
    echo "ERROR: Could not download dataset from any source."
    echo "Please manually download the Intel Research Lab dataset and place it at:"
    echo "  $DATA_DIR/intel.gfs"
    exit 1
fi

# Verify the file has content
FILE_SIZE=$(stat -f%z "$DATA_DIR/intel.gfs" 2>/dev/null || stat -c%s "$DATA_DIR/intel.gfs" 2>/dev/null || echo 0)
if [ "$FILE_SIZE" -lt 1000 ]; then
    echo "ERROR: Downloaded file is too small ($FILE_SIZE bytes). May be corrupted."
    rm -f "$DATA_DIR/intel.gfs"
    exit 1
fi

# Count FLASER lines
FLASER_COUNT=$(grep -c "^FLASER\|^ROBOTLASER" "$DATA_DIR/intel.gfs" 2>/dev/null || echo 0)
echo "Dataset downloaded successfully."
echo "  File: $DATA_DIR/intel.gfs"
echo "  Size: $FILE_SIZE bytes"
echo "  Laser scans: $FLASER_COUNT"
