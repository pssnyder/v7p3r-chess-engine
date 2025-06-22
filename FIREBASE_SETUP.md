# Firebase Backend Setup

V7P3R Chess Engine now uses Firebase for its backend storage and data processing needs. This guide explains how to set up and use the Firebase backend for development and deployment.

## Prerequisites

1. Install Firebase CLI tools:

   ```bash
   npm install -g firebase-tools
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This includes the firebase-admin package required for accessing Firebase services.

## Firebase Setup

1. Log in to Firebase CLI:

   ```bash
   firebase login
   ```

2. Use our existing project:

   ```bash
   firebase use v7p3r-chess-engine
   ```

3. Download the service account key:
   - Visit the [Firebase Console](https://console.firebase.google.com/project/v7p3r-chess-engine/settings/serviceaccounts/adminsdk)
   - Click "Generate new private key"
   - Save the downloaded JSON file as `firebase_service_account.json` in the `config` directory

## Firebase Services Used

- **Firestore Database**: Stores game metadata, metrics, and model information
- **Cloud Storage**: Stores PGN files, trained models, and other artifacts
- **Firebase Authentication**: Manages user accounts (optional for local development)

## Directory Structure

- `config/firebase_config.py`: Firebase configuration and initialization
- `engine_utilities/firebase_cloud_store.py`: Firebase API client for V7P3R
- `engine_utilities/cloud_store.py`: Original Google Cloud Storage client (maintained for backward compatibility)

## Local Development with Emulators

For local development without affecting production data:

1. Start the Firebase emulators:

   ```bash
   firebase emulators:start
   ```

2. Set environment variable to use emulators:

   ```bash
   $env:FIREBASE_USE_EMULATOR="true"  # PowerShell
   ```

## Data Schema

### Firestore Collections

- **games**: Game data and metadata
- **metrics**: Engine performance metrics
- **models**: Model metadata and version tracking
- **users**: User profiles and settings (when auth is enabled)

### Cloud Storage Structure

- **pgns/**: PGN files of games
- **models/**: Trained ML models with versioning
- **evaluations/**: Evaluation data
- **archives/**: Historical data for long-term storage

## Integration with V7P3R Components

The Firebase backend integrates with:

1. **Training Pipeline**: Stores model checkpoints, metrics, and configuration
2. **Game Engine**: Records game PGNs and metrics
3. **Analytics**: Provides data for performance analysis and visualization
4. **Web Interface**: Authenticates users and displays stored games/analysis

## Security Rules

Firebase security rules are defined in:

- `firestore.rules`: Controls access to Firestore data

- `storage.rules`: Controls access to Cloud Storage files

When deploying changes to these rules, use:

```bash
firebase deploy --only firestore:rules,storage
```
