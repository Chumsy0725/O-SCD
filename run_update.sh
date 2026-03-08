#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate FOSCD

# Define instances and classes
INSTANCES=("Instance_1" "Instance_2")
# INSTANCES=("Instance_1")

CLASSES=("Cantina" "Garden" "Lounge" "Lunch_room" "Meeting_room" "Playground" "Porch" "Pots" "Printing_area" "Zen")
# CLASSES=("Garden")



# Base paths
DATA_ROOT="data/PASLCD"
OUTPUT_ROOT="output"

# Set to true to run with test hold and evaluation
TESTING=false

for INSTANCE_NAME in "${INSTANCES[@]}"; do
    for CLASSNAME in "${CLASSES[@]}"; do
        echo "Running Online Scene Change Detection: $INSTANCE_NAME, $CLASSNAME"
        
        if [ "$TESTING" = true ]; then
            # Run OSCD with test hold
            CMD="python update.py -s \"$DATA_ROOT/$INSTANCE_NAME/$CLASSNAME/\" -m \"$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/\" --resolution 4 --test_hold 5"
            eval $CMD
            
            # Run Metrics
            python utils/metrics.py -m "$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/"
        else
            # Run OSCD without test hold
            CMD="python update.py -s \"$DATA_ROOT/$INSTANCE_NAME/$CLASSNAME/\" -m \"$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/\" --resolution 4"
            eval $CMD
        fi
 
    done
done
