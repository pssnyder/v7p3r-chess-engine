#!/usr/bin/env python3
"""
Package ChessBot as Windows executable using PyInstaller
"""

import subprocess
import sys
import os
import shutil

def install_pyinstaller():
    """Install PyInstaller if not present"""
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
        return True
    except ImportError:
        print("📦 Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
            print("✅ PyInstaller installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyInstaller: {e}")
            return False

def package_uci_engine():
    """Package UCI engine as executable"""
    print("\n📦 Packaging UCI Engine...")

    cmd = [
        'pyinstaller',
        '--onefile',
        '--name=ChessBot_UCI',
        '--add-data=evaluation_engine.py:.',
        '--add-data=main_engine.py:.',
        '--add-data=time_manager.py:.',
        '--add-data=config.py:.',
        '--hidden-import=chess',
        '--hidden-import=numpy',
        '--console',
        'uci_interface.py'
    ]

    try:
        subprocess.check_call(cmd)
        print("✅ UCI Engine packaged successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to package UCI engine: {e}")
        return False

def package_lichess_bot():
    """Package Lichess bot as executable"""
    print("\n📦 Packaging Lichess Bot...")

    cmd = [
        'pyinstaller',
        '--onefile',
        '--name=ChessBot_Lichess',
        '--add-data=evaluation_engine.py:.',
        '--add-data=main_engine.py:.',
        '--add-data=time_manager.py:.',
        '--add-data=config.py:.',
        '--hidden-import=chess',
        '--hidden-import=numpy',
        '--hidden-import=requests',
        '--console',
        'lichess_bot.py'
    ]

    try:
        subprocess.check_call(cmd)
        print("✅ Lichess Bot packaged successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to package Lichess bot: {e}")
        return False

def cleanup():
    """Clean up build files"""
    print("\n🧹 Cleaning up build files...")

    dirs_to_remove = ['build', '__pycache__']
    files_to_remove = ['*.spec']

    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")

    # Remove .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))

def main():
    """Main packaging function"""
    print("📦 ChessBot Packaging")
    print("=" * 30)

    if not install_pyinstaller():
        sys.exit(1)

    success = True

    if package_uci_engine():
        print("✅ UCI engine executable created in dist/ChessBot_UCI.exe")
    else:
        success = False

    if package_lichess_bot():
        print("✅ Lichess bot executable created in dist/ChessBot_Lichess.exe")
    else:
        success = False

    if success:
        print("\n🎉 Packaging complete!")
        print("\nExecutables created:")
        print("- dist/ChessBot_UCI.exe (for UCI GUIs like Arena)")
        print("- dist/ChessBot_Lichess.exe (for Lichess bot)")
        print("\nUsage:")
        print("- UCI: Load ChessBot_UCI.exe in your chess GUI")
        print("- Lichess: Run ChessBot_Lichess.exe <your_token>")
    else:
        print("\n❌ Packaging failed. Check error messages above.")

    cleanup()

if __name__ == "__main__":
    main()
