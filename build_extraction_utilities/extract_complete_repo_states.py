# Complete Repository State Extraction
# This script extracts the FULL repository state at each beta candidate commit point

import os
import subprocess
import shutil

def extract_complete_repo_state(tag, target_dir):
    """Extract complete repository state at a specific tag"""
    
    print(f"Extracting complete repo state for {tag}...")
    
    # Create temporary directory for full extraction
    temp_dir = f"temp_extraction_{tag}"
    
    try:
        # Remove existing target directory
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # Create fresh directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Get the commit hash for this tag
        result = subprocess.run(['git', 'rev-list', '-n', '1', tag], 
                              capture_output=True, text=True, check=True)
        commit_hash = result.stdout.strip()
        
        print(f"  Commit hash: {commit_hash}")
        
        # Get ALL files that existed at this commit (not just changed files)
        result = subprocess.run(['git', 'ls-tree', '-r', '--name-only', commit_hash], 
                              capture_output=True, text=True, check=True)
        all_files = result.stdout.strip().split('\n')
        
        print(f"  Found {len(all_files)} total files in repository at this point")
        
        # Filter for relevant files
        relevant_extensions = {'.py', '.json', '.yaml', '.yml', '.md', '.db', '.txt'}
        relevant_files = []
        
        for file_path in all_files:
            if file_path and any(file_path.endswith(ext) for ext in relevant_extensions):
                # Skip some obvious non-essential files
                if not any(skip in file_path.lower() for skip in ['__pycache__', '.git', 'node_modules', '.vscode']):
                    relevant_files.append(file_path)
        
        print(f"  Extracting {len(relevant_files)} relevant files...")
        
        # Extract each file
        extracted_count = 0
        failed_count = 0
        
        for file_path in relevant_files:
            try:
                # Get file content at this commit
                result = subprocess.run(['git', 'show', f'{commit_hash}:{file_path}'], 
                                      capture_output=True, text=True, check=True)
                file_content = result.stdout
                
                # Create target path
                target_file_path = os.path.join(target_dir, file_path)
                target_file_dir = os.path.dirname(target_file_path)
                
                # Create directory structure
                if target_file_dir and not os.path.exists(target_file_dir):
                    os.makedirs(target_file_dir, exist_ok=True)
                
                # Write file
                with open(target_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                
                extracted_count += 1
                if extracted_count % 10 == 0:
                    print(f"    Extracted {extracted_count}/{len(relevant_files)} files...")
                    
            except subprocess.CalledProcessError:
                failed_count += 1
                print(f"    ⚠️ Failed to extract: {file_path}")
            except Exception as e:
                failed_count += 1
                print(f"    ⚠️ Error extracting {file_path}: {e}")
        
        print(f"  ✅ Extraction complete: {extracted_count} successful, {failed_count} failed")
        
        # Create extraction info
        info = {
            'tag': tag,
            'commit_hash': commit_hash,
            'total_files_in_repo': len(all_files),
            'relevant_files_found': len(relevant_files),
            'files_extracted': extracted_count,
            'files_failed': failed_count,
            'extraction_timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip()
        }
        
        import json
        with open(os.path.join(target_dir, 'EXTRACTION_INFO.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to extract {tag}: {e}")
        return False

def main():
    """Extract complete repository states for all beta candidates"""
    
    # Beta candidates in chronological order
    beta_candidates = [
        'beta-candidate-16',  # 2025-06-01 - earliest
        'beta-candidate-15',  # 2025-06-02
        'beta-candidate-14',  # 2025-06-04
        'beta-candidate-7',   # 2025-06-04
        'beta-candidate-13',  # 2025-06-05
        'beta-candidate-12',  # 2025-06-07
        'beta-candidate-11',  # 2025-06-09
        'beta-candidate-4',   # 2025-06-09
        'beta-candidate-5',   # 2025-05-31
        'beta-candidate-6',   # 2025-05-30
        'beta-candidate-3',   # 2025-06-27 - THIS ONE specifically
        'beta-candidate-10',  # 2025-06-30
        'beta-candidate-2',   # 2025-07-01
        'beta-candidate-1',   # 2025-07-03
        'beta-candidate-9',   # 2025-07-07
        'beta-candidate-8',   # 2025-07-14
        'beta-candidate-0',   # 2025-07-15 - latest
    ]
    
    print("🔄 EXTRACTING COMPLETE REPOSITORY STATES")
    print("=" * 60)
    
    # Create new complete builds directory
    complete_builds_dir = "builds_complete"
    if os.path.exists(complete_builds_dir):
        print(f"Removing existing {complete_builds_dir}...")
        shutil.rmtree(complete_builds_dir)
    
    os.makedirs(complete_builds_dir, exist_ok=True)
    
    successful_extractions = []
    failed_extractions = []
    
    for tag in beta_candidates:
        print(f"\n📦 Processing {tag}...")
        
        # Get date for directory naming
        try:
            result = subprocess.run(['git', 'show', '-s', '--format=%ci', tag], 
                                  capture_output=True, text=True, check=True)
            date_str = result.stdout.strip().split()[0]  # Get just the date part
            
            # Convert to version format
            date_parts = date_str.split('-')
            month = int(date_parts[1])
            day = int(date_parts[2])
            version_name = f"v0.{month}.{day}_{tag}"
            
            target_dir = os.path.join(complete_builds_dir, version_name)
            
            success = extract_complete_repo_state(tag, target_dir)
            
            if success:
                successful_extractions.append((tag, version_name))
            else:
                failed_extractions.append(tag)
                
        except Exception as e:
            print(f"❌ Error processing {tag}: {e}")
            failed_extractions.append(tag)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(successful_extractions)}")
    for tag, version_name in successful_extractions:
        print(f"   - {tag} → {version_name}")
    
    if failed_extractions:
        print(f"\n❌ Failed: {len(failed_extractions)}")
        for tag in failed_extractions:
            print(f"   - {tag}")
    
    print(f"\n📁 Complete builds available in: {complete_builds_dir}/")
    print("\nEach build now contains the COMPLETE repository state at that point in time!")

if __name__ == "__main__":
    main()
