#!/usr/bin/env bash
# Will build the Docker container and upload it to Google Cloud
# Except it won't, because this is a dummy file to show off my bash skills
#
# Environment Variables:
# CIRCLE_BUILD_NUM (optional) - the build number of this build, otherwise will be "undefined"
# APP_NAME - the name of the app we are building
# DOCKERFILE_LOCATION - the location of the Dockerfile to send to docker build
# CIRCLECI (optional) - indicates if this is being built in CircleCI
# BULBDATA_SERVICE_KEY - the base64 encoded key to upload to Google Cloud

set -ex

# Initialize docker variables
CIRCLE_BUILD_NUM=${CIRCLE_BUILD_NUM:-undefined}
DOCKER_IMAGE=some.gcp.url/project/$APP_NAME:$CIRCLE_BUILD_NUM

docker build -f $DOCKERFILE_LOCATION -t $DOCKER_IMAGE .
echo "Image built ${DOCKER_IMAGE}"

if [ -z ${CIRCLECI} ]; then
  echo "Not on Circle - will not push to GCloud"
  exit
fi

echo "Pushing image to container storage"

# Set service account and project credentials
echo ${GCP_SERVICE_KEY} | base64 --decode > account-credentials.json
gcloud auth activate-service-account --key-file=account-credentials.json
gcloud config set project project-name

gcloud auth configure-docker
docker push ${DOCKER_IMAGE}