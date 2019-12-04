# This bash script deploys our docker image containing the latest version of our model to k8s
# Except it doesn't, because this is a dummy script
#!/bin/bash
set -e

die() {
  echo "$1"
  exit 1
}

GCP_KEY=${GCP_SERVICE_KEY}
[ -z ${GCP_KEY} ] && die "😦 🆘 Must provide BULBENRG_SERVICE_KEY"
NAMESPACE=${NAMESPACE}
[ -z ${NAMESPACE} ] && die "😦 🆘 Must provide kubernetes namespace"

# Initalise vars
BUILD_NUMBER=${CIRCLE_BUILD_NUM}
DEPLOYMENT_NAME=${APP_NAME/-service/}

# Set gcloud vars
GCLOUD_CLUSTER=${GCLOUD_CLUSTER:-"cluster42"}
GCLOUD_ZONE=${GCLOUD_ZONE:-"europe-west1-c"}
GCLOUD_PROJECT=${GCLOUD_PROJECT:-"project-name"}

# Authenticate account
echo ${GCP_KEY} | base64 --decode > account-credentials.json
gcloud auth activate-service-account --key-file=account-credentials.json
gcloud container clusters get-credentials ${GCLOUD_CLUSTER} --zone ${GCLOUD_ZONE} --project ${GCLOUD_PROJECT}

# Deploy to k8s
echo "Deploying ${APP_NAME}:${BUILD_NUMBER} on ${NAMESPACE} namespace..."
echo "Updating k8s manifest with built image ${BUILD_NUMBER}"
sed -i".bak" -E 's|(eu.gcr.io/[^/]+/'"$APP_NAME"':)([0-9])+|eu.gcr.io/project-name/'"$APP_NAME"':'"$BUILD_NUMBER"'|g' ./k8s/manifest.yml
echo "Applying updated manifest 📃"

kubectl --namespace="${NAMESPACE}" apply -f ./k8s/manifest.yml
kubectl --namespace="${NAMESPACE}" rollout status deploy "${DEPLOYMENT_NAME}"

echo "${APP_NAME}:${BUILD_NUMBER} has been rolled out on ${NAMESPACE}/${DEPLOYMENT_NAME}! 🛳"