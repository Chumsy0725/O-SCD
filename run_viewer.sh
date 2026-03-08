#!/bin/bash

# run_viewer.sh
# Usage: ./run_viewer.sh <Scene_Name> <Instance_Number> [Port]
# Example: ./run_viewer.sh Lounge 1 8080

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <Scene_Name> <Instance_Number> [Port]"
    echo "Example: $0 Lounge 1 8080"
    exit 1
fi

SCENE="$1"
INSTANCE="$2"
PORT="${3:-8080}"

# Base paths based on the project structure
DATA_DIR="/home/chamuditha/Desktop/CVPR26/OSCD/data/PASLCD"
OUTPUT_DIR="/home/chamuditha/Desktop/CVPR26/OSCD/output"

# Construct paths
SOURCE_PATH="${DATA_DIR}/Instance_${INSTANCE}/${SCENE}"
REF_PLY="${SOURCE_PATH}/reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
INFERENCE_DIR="${SOURCE_PATH}/inference_scene/images"

MODEL_PATH="${OUTPUT_DIR}/Instance_${INSTANCE}/${SCENE}"
UPDATED_PLY="${MODEL_PATH}/updated_scene.ply"
CHANGE_PLY="${MODEL_PATH}/change.ply"
CAMERAS_JSON="${MODEL_PATH}/cameras.json"

echo "========================================="
echo "Launching Viser Viewer"
echo "Scene: ${SCENE}"
echo "Instance: ${INSTANCE}"
echo "Port: ${PORT}"
echo "========================================="
echo "Reference PLY: ${REF_PLY}"
echo "Updated PLY:   ${UPDATED_PLY}"
echo "Change PLY:    ${CHANGE_PLY}"
echo "Cameras JSON:  ${CAMERAS_JSON}"
echo "========================================="

# Check if paths exist
if [ ! -f "$REF_PLY" ]; then
    echo "Warning: Reference PLY not found at $REF_PLY"
fi

if [ ! -d "$INFERENCE_DIR" ]; then
    echo "Warning: Inference directory not found at $INFERENCE_DIR"
fi

# We don't exit if updated/change plies are missing since the viewer can handle them optionally,
# but usually we want them to exist.

# Run the viewer
python viewer.py \
    --ref_ply "$REF_PLY" \
    --updated_ply "$UPDATED_PLY" \
    --change_ply "$CHANGE_PLY" \
    --cameras_json "$CAMERAS_JSON" \
    --inference_dir "$INFERENCE_DIR" \
    --port "$PORT"
