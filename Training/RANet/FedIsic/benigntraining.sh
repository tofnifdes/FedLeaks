#!/bin/bash

echo "Starting server"
python3 server.py --attribute None &
sleep 3  # Sleep for 3s to give the server enough time to start
for (( i=0 ; i<6 ; i++ ));  do
    echo "Starting benign client $i"
    python3 benign_client.py --partition-id $i &

done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
