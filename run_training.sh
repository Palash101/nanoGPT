#!/bin/bash
# Script to run training in the background with logging

# Configuration
CONFIG="config/train_tinystories_medium.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Run training in background with nohup
echo "Starting training..."
echo "Log file: ${LOG_FILE}"
echo "To view logs: tail -f ${LOG_FILE}"
echo "To check if running: ps aux | grep train.py"

nohup python train.py ${CONFIG} > ${LOG_FILE} 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: ${PID}"
echo "PID saved to: ${LOG_DIR}/training_${TIMESTAMP}.pid"
echo ${PID} > ${LOG_DIR}/training_${TIMESTAMP}.pid

echo ""
echo "To stop training: kill ${PID}"
echo "To monitor: tail -f ${LOG_FILE}"


