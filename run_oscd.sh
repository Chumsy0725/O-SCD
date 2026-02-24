#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oscd

# Define instances and classes
INSTANCES=("Instance_1" "Instance_2")
# INSTANCES=("Instance_2")

CLASSES=("Cantina" "Garden" "Lounge" "Lunch_room" "Meeting_room" "Playground" "Porch" "Pots" "Printing_area" "Zen")
# CLASSES=("Garden")

# Refine flag - set to true to offline post refinement
REFINE=true

# Base paths
DATA_ROOT="data/PASLCD"
OUTPUT_ROOT="output/"

for INSTANCE_NAME in "${INSTANCES[@]}"; do
    for CLASSNAME in "${CLASSES[@]}"; do
        echo "Running Online Scene Change Detection: $INSTANCE_NAME, $CLASSNAME"
        
        # Run OSCD (Uncomment to run)
        CMD="python oscd.py -s \"$DATA_ROOT/$INSTANCE_NAME/$CLASSNAME/\" -m \"$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/\" --resolution 4 --test_hold 5"
        if [ "$REFINE" = true ]; then
            CMD="$CMD --refine"
        fi
        eval $CMD
 
        # Evaluate Change Masks
        echo "Evaluating Change Masks..."
        python utils/evaluate.py --gt "$DATA_ROOT/$INSTANCE_NAME/$CLASSNAME/gt_mask/" --pred_binary "$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/renders/change_mask/" > "$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/evaluation.txt"
       

        # Evaluate Refined Masks
        if [ "$REFINE" = true ]; then
            echo "Evaluating Refined Masks..."
            python utils/evaluate.py --gt "$DATA_ROOT/$INSTANCE_NAME/$CLASSNAME/gt_mask/" --pred_binary "$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/renders/change_mask_refined/" > "$OUTPUT_ROOT/$INSTANCE_NAME/$CLASSNAME/evaluation_refined.txt"
        fi 
    done
done
