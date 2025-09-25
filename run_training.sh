#!/bin/bash

# Configuration
TIMEOUT=900  # 15 minutes in seconds
SCRIPT_TO_RUN="$1"  # Pass your script as the first argument
TEMP_LOG="/tmp/script_output_$$.log"
LAST_UPDATE_FILE="/tmp/last_update_$$.tmp"

# Check if script was provided
if [ -z "$SCRIPT_TO_RUN" ]; then
  echo "Usage: $0 <script_to_monitor>"
  exit 1
fi

# Function to clean up before exit
cleanup() {
  echo "Shutting down monitored script..."
  # Kill all processes in the script's process group
  if [ ! -z "$PGID" ]; then
    kill -TERM -$PGID 2>/dev/null
  fi
  
  # Remove temporary files
  rm -f "$TEMP_LOG" "$LAST_UPDATE_FILE"
  
  exit 0
}

# Handle Ctrl+C and other signals
trap cleanup SIGINT SIGTERM SIGHUP

# Main monitoring loop
while true; do
  # Reset output log and update time tracker
  > "$TEMP_LOG"
  touch "$LAST_UPDATE_FILE"
  
  # Create a named pipe for output capture
  FIFO_OUT="/tmp/script_fifo_out_$$.pipe"
  rm -f "$FIFO_OUT"
  mkfifo "$FIFO_OUT"
  
  # Start background process to read from the FIFO, tee to both the terminal and our log file
  # This ensures we don't block the fifo writer
  cat "$FIFO_OUT" | tee -a "$TEMP_LOG" &
  TEE_PID=$!
  
  # Start the script in its own process group with all output redirected to our fifo
  # Using stdbuf to disable buffering which can cause output delays
  (
    # Set up new process group
    exec setsid bash -c "
      # Use stdbuf to ensure unbuffered output
      stdbuf -i0 -o0 -e0 \"$SCRIPT_TO_RUN\" 
    " 
  ) > "$FIFO_OUT" 2>&1 &
  
  SCRIPT_PID=$!
  # Get the process group ID for properly terminating all children
  PGID=$(ps -o pgid= $SCRIPT_PID | tr -d ' ')
  
  echo "Started script (PID: $SCRIPT_PID, PGID: $PGID)"

  # Monitor script output
  while kill -0 $SCRIPT_PID 2>/dev/null; do
    # Check if there has been new output
    if [ -s "$TEMP_LOG" ] && [ "$(stat -c %Y "$TEMP_LOG")" -gt "$(stat -c %Y "$LAST_UPDATE_FILE")" ]; then
      touch "$LAST_UPDATE_FILE"
    fi
    
    # Check if output has stalled
    current_time=$(date +%s)
    last_update=$(stat -c %Y "$LAST_UPDATE_FILE")
    elapsed=$((current_time - last_update))
    
    if [ $elapsed -gt $TIMEOUT ]; then
      echo "No output for ${TIMEOUT}s - Restarting script"
      # Kill the entire process group
      kill -TERM -$PGID 2>/dev/null
      wait $SCRIPT_PID 2>/dev/null
      break
    fi
    
    sleep 1
  done
  
  # Clean up the tee process and fifo
  kill $TEE_PID 2>/dev/null
  wait $TEE_PID 2>/dev/null
  rm -f "$FIFO_OUT"
  
  # If we're here and the script has exited normally, check the exit code
  wait $SCRIPT_PID
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully (Exit code: $EXIT_CODE)"
    cleanup
  else
    echo "Script exited with code $EXIT_CODE, restarting in 3 seconds..."
    sleep 3
  fi
done