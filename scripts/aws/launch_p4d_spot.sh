#!/usr/bin/env bash
# launch_p4d_spot.sh — Request a p4d.24xlarge spot instance via AWS CLI
# Usage: ./launch_p4d_spot.sh [--region us-east-1] [--az us-east-1a]
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure or instance role)
#   - A key pair created in the target region
#   - A security group allowing SSH (port 22) from your IP
#   - An S3 bucket for results/checkpoints

set -euo pipefail

# ── Configuration — edit these ────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
AZ="${AZ:-us-east-1a}"                          # Availability zone for spot
KEY_NAME="${KEY_NAME:-your-key-pair-name}"       # EC2 key pair name
SECURITY_GROUP="${SG_ID:-sg-xxxxxxxxxxxxxxxxx}"  # Security group ID
SUBNET_ID="${SUBNET_ID:-subnet-xxxxxxxxxxxxxxxxx}"
# Deep Learning Base GPU AMI (Ubuntu 22.04) — update for your region/date
# Find latest: aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning Base GPU AMI (Ubuntu 22.04)*" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId'
AMI_ID="${AMI_ID:-ami-xxxxxxxxxxxxxxxxx}"
INSTANCE_TYPE="p4d.24xlarge"
SPOT_PRICE="10.00"                               # Max price (well above ~$4.41 to avoid interruption)
VOLUME_SIZE_GB=200                               # Root EBS volume
S3_BUCKET="${S3_BUCKET:-your-results-bucket}"

# ── Request spot instance ─────────────────────────────────────────────────────
echo "Requesting ${INSTANCE_TYPE} spot in ${AZ} (max \$${SPOT_PRICE}/hr)..."

SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --region "${REGION}" \
    --spot-price "${SPOT_PRICE}" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"${AMI_ID}\",
        \"InstanceType\": \"${INSTANCE_TYPE}\",
        \"KeyName\": \"${KEY_NAME}\",
        \"SecurityGroupIds\": [\"${SECURITY_GROUP}\"],
        \"SubnetId\": \"${SUBNET_ID}\",
        \"Placement\": {\"AvailabilityZone\": \"${AZ}\"},
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {
                \"VolumeSize\": ${VOLUME_SIZE_GB},
                \"VolumeType\": \"gp3\",
                \"DeleteOnTermination\": true
            }
        }],
        \"UserData\": \"$(base64 -w0 <<'USERDATA'
#!/bin/bash
# Auto-setup on first boot
apt-get update -qq
# aws-ofi-nccl and EFA should already be on DL AMI
# Verify NCCL
python3 -c "import subprocess; subprocess.run(['nvcc','--version'])"
USERDATA
)\"
    }")

REQUEST_ID=$(echo "${SPOT_REQUEST}" | python3 -c "import sys,json; print(json.load(sys.stdin)['SpotInstanceRequests'][0]['SpotInstanceRequestId'])")
echo "Spot request ID: ${REQUEST_ID}"

# ── Wait for fulfillment ──────────────────────────────────────────────────────
echo "Waiting for spot instance to be fulfilled..."
aws ec2 wait spot-instance-request-fulfilled \
    --region "${REGION}" \
    --spot-instance-request-ids "${REQUEST_ID}"

INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --region "${REGION}" \
    --spot-instance-request-ids "${REQUEST_ID}" \
    --query 'SpotInstanceRequests[0].InstanceId' \
    --output text)

echo "Instance launched: ${INSTANCE_ID}"
echo "Waiting for status checks..."
aws ec2 wait instance-status-ok --region "${REGION}" --instance-ids "${INSTANCE_ID}"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "${REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Instance ready ==="
echo "  Instance ID : ${INSTANCE_ID}"
echo "  Public IP   : ${PUBLIC_IP}"
echo "  SSH         : ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "  S3 bucket   : s3://${S3_BUCKET}/"
echo ""
echo "Next step: run scripts/aws/setup_p4d_env.sh on the instance"
echo "To terminate: aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
