# V7P3R Labs Deployment Guide

## Quick Start Deployment (Coming-Soon Page)

The coming-soon placeholder is ready to deploy **right now** without installing any dependencies!

### Prerequisites
- Firebase CLI installed (`npm install -g firebase-tools`)
- Authenticated with Firebase (`firebase login`)
- Firebase project `rts-labs-f3981` access

### Deploy Coming-Soon Page NOW

```powershell
# Navigate to v7p3r-labs directory
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r-labs"

# Ensure dist folder has the coming-soon page
Copy-Item index.html dist\index.html

# Deploy to Firebase
firebase deploy --only hosting:v7p3r-labs
```

**Live URLs after deployment:**
- Custom Domain: https://v7p3r.labs.rtsts.tech
- Firebase URL: https://v7p3r-chess-engine-f3981.web.app

---

## Full Development Deployment (Future React App)

Once the React application is built, use this workflow:

### 1. Install Dependencies
```powershell
npm install
```

### 2. Build for Production
```powershell
npm run build
```

This will:
- Compile TypeScript
- Bundle React application with Vite
- Output to `dist/` directory

### 3. Deploy to Firebase
```powershell
npm run deploy
```

Or use Firebase CLI directly:
```powershell
firebase deploy --only hosting:v7p3r-labs
```

---

## GitHub Actions CI/CD Workflow

The repository is configured with automatic deployments via GitHub Actions.

**Workflow File:** `.github/workflows/deploy-v7p3r-labs.yml`

### Triggers
- Push to `main` branch with changes in `v7p3r-labs/**`
- Manual workflow dispatch

### Requirements
GitHub Secrets needed:
- `FIREBASE_SERVICE_ACCOUNT_RTS_LABS` - Firebase service account JSON

### How It Works
1. Checkout code
2. Install Node.js 24
3. Guard against compromised packages
4. Install dependencies (`npm ci`)
5. Build frontend (`npm run build`)
6. Deploy to Firebase Hosting (target: `v7p3r-labs`)

**To trigger manual deployment:**
1. Go to GitHub Actions tab
2. Select "Deploy V7P3R Labs to Production"
3. Click "Run workflow"

---

## Firebase Configuration

### Hosting Target
Configured in `.firebaserc`:
```json
{
  "targets": {
    "rts-labs-f3981": {
      "hosting": {
        "v7p3r-labs": ["v7p3r-chess-engine-f3981"]
      }
    }
  }
}
```

### Hosting Settings
Configured in `firebase.json`:
- Public directory: `dist/`
- SPA routing: All routes → `/index.html`
- Cache headers: 1 year for static assets (images, JS, CSS)

---

## Development Workflow

### Local Development
```powershell
# Start Vite dev server
npm run dev
```

Access at: http://localhost:5173

### Preview Production Build
```powershell
# Build and preview
npm run build
npm run preview
```

### Firebase Hosting Preview (Staging)
```powershell
# Deploy to preview channel
firebase hosting:channel:deploy preview --expires 7d
```

---

## Troubleshooting

### Deployment Fails
1. Verify Firebase authentication: `firebase login`
2. Check project ID: `firebase use rts-labs-f3981`
3. Verify hosting target exists: `firebase target:apply hosting v7p3r-labs v7p3r-chess-engine-f3981`

### Custom Domain Not Working
1. Verify DNS settings in domain registrar
2. Check Firebase Console → Hosting → Custom Domains
3. Domain propagation can take 24-48 hours

### Build Errors
1. Clear node_modules: `Remove-Item -Recurse -Force node_modules`
2. Clear dist: `Remove-Item -Recurse -Force dist`
3. Reinstall: `npm install`
4. Rebuild: `npm run build`

---

## Project Structure

```
v7p3r-labs/
├── .firebaserc          # Firebase project config
├── firebase.json        # Firebase hosting settings
├── package.json         # Node dependencies & scripts
├── vite.config.ts       # Vite bundler config
├── tsconfig.json        # TypeScript config
├── tailwind.config.js   # Tailwind CSS config
├── postcss.config.js    # PostCSS config
├── index.html           # Entry point (currently coming-soon)
├── coming-soon.html     # Original coming-soon template
├── dist/                # Build output (deployed to Firebase)
├── src/                 # Future React application code
└── images/              # Custom chess piece images
```

---

## Next Steps

1. ✅ Deploy coming-soon placeholder
2. Build React frontend with chessboard UI
3. Implement backend architecture (Option 1 or Option 2)
4. Integrate UCI engine communication
5. Test and iterate
6. Replace coming-soon with full application
7. Update custom domain DNS if needed

---

**Last Updated:** April 17, 2026  
**Maintainer:** Patrick Snyder  
**Project:** V7P3R Chess Engine @ RTS Labs
