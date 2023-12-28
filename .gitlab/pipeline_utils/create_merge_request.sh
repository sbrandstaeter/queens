#!/usr/bin/bash

# Create a new branch, add desired changes and create an MR

# Some logging
echo Creating Branch
echo - NEW_BRANCH_NAME $NEW_BRANCH_NAME
echo - FILES $FILES # files to commit
echo - COMMIT_MESSAGE $COMMIT_MESSAGE

echo Creating merge request
echo - MR_TITLE $MR_TITLE

# Go to pulled repo
cd $CI_PROJECT_DIR

# Check for changes in the desired files
CHANGES=$(git status $FILES --porcelain)
if [ -z "${CHANGES}" ]; then
  # Ending job, as the files did not change
  echo "Nothing changed, not creating an MR."
  exit 0
fi

# Basic git config
GITLAB_USER_NAME=project_${CI_PROJECT_ID}_bot
git config --global user.name "${AUTO_COMMITTER_NAME}"
git config --global user.email "${AUTO_COMMITTER_EMAIL}"

# Checkout to a new branch so we push the changes to a new branch
git branch -f $NEW_BRANCH_NAME && git checkout $NEW_BRANCH_NAME

# Add the files and commit
git add $FILES
git commit -m "${COMMIT_MESSAGE}"

# Push branch
git push --force "https://${GITLAB_USER_NAME}:${GIT_PUSH_TOKEN}@${CI_REPOSITORY_URL#*@}"

# Open MR
BODY="{\"id\": \"${CI_PROJECT_ID}\",
       \"source_branch\": \"${NEW_BRANCH_NAME}\",
       \"target_branch\": \"${CI_DEFAULT_BRANCH}\",
       \"remove_source_branch\": true,
       \"title\": \"${MR_TITLE}\"}"

MR_IID=$(curl -sS -X POST "https://gitlab.lrz.de/api/v4/projects/${CI_PROJECT_ID}/merge_requests" \
        --header "PRIVATE-TOKEN:${GIT_PUSH_TOKEN}" \
        --header "Content-Type: application/json" \
        --data "${BODY}" \
| jq '.iid')
echo "Opened MR !${MR_IID}: ${MR_TITLE}"
