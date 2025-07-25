#!/usr/bin/env python3
"""
Setup script for ChessBot
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True

def setup_lichess_token():
    """Help user set up Lichess token"""
    print("\n📝 Setting up Lichess API Token:")
    print("1. Go to https://lichess.org/account/oauth/token")
    print("2. Create a new token with 'Bot Play' scope")
    print("3. Copy the token")

    token = input("\nPaste your Lichess token here (or press Enter to skip): ").strip()

    if token:
        # Save token to environment file
        with open('.env', 'w') as f:
            f.write(f"LICHESS_TOKEN={token}\n")
        print("✅ Token saved to .env file")

        # Upgrade account to bot
        print("\n🔄 Upgrading Lichess account to bot...")
        print(f"Run this command in your terminal:")
        print(f'curl -d "" https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer {token}"')
        print("\nOr the bot will automatically upgrade your account when you first run it.")
    else:
        print("⚠️  You can set up the token later using environment variables")

def main():
    """Main setup function"""
    print("🏁 ChessBot Setup")
    print("=" * 30)

    if not check_python_version():
        sys.exit(1)

    if not install_requirements():
        sys.exit(1)

    setup_lichess_token()

    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Test the engine: python uci_interface.py")
    print("2. Run Lichess bot: python lichess_bot.py")
    print("3. Package as exe: python package_exe.py")

if __name__ == "__main__":
    main()
