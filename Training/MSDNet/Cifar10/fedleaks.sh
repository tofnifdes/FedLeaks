#!/bin/bash

echo "Starting server"
python3 server.py --attribute classLabel &
sleep 3  # Sleep for 3s to give the server enough time to start
echo "Starting malicious client 0"
python3 malicious_client.py --partition-id 0 &
for (( i=1 ; i<10 ; i++ ));  do
    echo "Starting benign client $i"
    python3 benign_client.py --partition-id $i &

done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
