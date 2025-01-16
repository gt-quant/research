# scp -i "/Users/giovanni/Development/GT Quant/ec2/Trading.pem" "ec2-user@ec2-18-136-204-54.ap-southeast-1.compute.amazonaws.com:/home/ec2-user/deploy/OB1/BTCUSDT/20241212.data.gz" data/OB1/BTCUSDT/

#!/bin/bash

# Parameters
SYMBOL=$1
DATE=$2
REMOTE_PATH="/home/ec2-user/deploy/OB1/$SYMBOL/$DATE.data.gz"
LOCAL_PATH="data/OB1/$SYMBOL/$DATE.data.gz"
LOCAL_DIR="data/OB1/$SYMBOL/"
PEM_PATH="/Users/giovanni/Development/GT Quant/ec2/Trading.pem"
EC2_USER="ec2-user"
EC2_HOST="ec2-18-136-204-54.ap-southeast-1.compute.amazonaws.com"

# Ensure both parameters are provided
if [ -z "$SYMBOL" ] || [ -z "$DATE" ]; then
  echo "Usage: $0 <SYMBOL> <DATE>"
  exit 1
fi

# Check if decompressed file already exists
UNCOMPRESSED_FILE="${LOCAL_PATH%.gz}"
if [ -f "$UNCOMPRESSED_FILE" ]; then
  echo "Uncompressed file already exists. Skipping SCP and deflation."
  exit 0
fi

# SCP command
if [ ! -f "$LOCAL_PATH" ]; then
  echo "Compressed file does not exist locally. Fetching from server..."
  scp -i "$PEM_PATH" "$EC2_USER@$EC2_HOST:$REMOTE_PATH" "$LOCAL_DIR"
else
  echo "Compressed file already exists locally. Skipping SCP."
fi

# Deflation command
if [ -f "$LOCAL_PATH" ]; then
  echo "Deflating the file..."
  gunzip "$LOCAL_PATH"
else
  echo "Compressed file not found. Unable to deflate."
fi
