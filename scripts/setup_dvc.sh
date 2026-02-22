#!/bin/bash

# Setup DVC for data versioning
# Run this script from the project root directory

echo "================================"
echo "Setting up DVC"
echo "================================"

# Initialize DVC
echo "Initializing DVC..."
dvc init

# Track data directory
echo "Adding data directory to DVC..."
dvc add data/processed

# Track models directory
echo "Adding models directory to DVC..."
dvc add models

echo ""
echo "DVC setup complete!"
echo ""
echo "Next steps:"
echo "1. Commit DVC files to Git:"
echo "   git add data/processed.dvc models.dvc .dvc/config .dvc/.gitignore .dvcignore"
echo "   git commit -m 'Initialize DVC tracking for data and models'"
echo ""
echo "2. (Optional) Configure remote storage:"
echo "   dvc remote add -d myremote s3://mybucket/dvcstore"
echo "   dvc remote modify myremote region us-east-1"
echo "   git add .dvc/config"
echo "   git commit -m 'Configure DVC remote storage'"
echo ""
echo "3. Push data to remote (if configured):"
echo "   dvc push"
