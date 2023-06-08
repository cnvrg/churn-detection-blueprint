#!/bin/bash

# Set the token pattern you want to search for
TOKEN_PATTERN="your_token_pattern_here"

# Get the commit hashes in reverse chronological order
COMMITS=$(git rev-list --all)

# Loop through each commit
for COMMIT in $COMMITS; do
    echo "Checking commit: $COMMIT"

    # Get the list of files changed in the commit
    FILES=$(git diff-tree --no-commit-id --name-only -r $COMMIT)

    # Loop through each changed file
    for FILE in $FILES; do
        # Check if the file contains the token pattern
        if grep -q "$TOKEN_PATTERN" "$FILE"; then
            echo "Token pattern found in file: $FILE"
        fi
    done

    echo
done
