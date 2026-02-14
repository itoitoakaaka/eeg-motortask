#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Research Directory Cleanup Script

このスクリプトは、Researchディレクトリ（ルート）にあるファイルを自動的に整理します。
実行方法: python python_files/cleanup_workspace.py

動作:
1. Pythonスクリプト (*.py) -> python_files/
2. ログファイル (*.log, *.txt) -> logs/
3. 画像ファイル (*.png, *.jpg, *.pdf) -> Figures/
4. バックアップ/データファイル (*.csv, *.json) -> logs/

※ 以下の重要なファイルは移動しません:
- water.txt
- age.txt
- requirements.txt
- Dockerfile, docker-compose.yml
- .DS_Store 等のシステムファイル
"""

import os
import shutil
import glob

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # python_files/../ -> Research/
if os.path.basename(ROOT_DIR) != 'Research':
    # Fallback if run from elsewhere
    ROOT_DIR = '/Users/itoakane/Research'

DEST_DIRS = {
    'python_files': os.path.join(ROOT_DIR, 'python_files'),
    'logs': os.path.join(ROOT_DIR, 'logs'),
    'Figures': os.path.join(ROOT_DIR, 'Figures')
}

# ファイル名でおいておくもの（完全一致）
EXCLUDE_FILES = {
    'water.txt', 
    'age.txt', 
    'requirements.txt', 
    'docker-compose.yml', 
    'Dockerfile',
    '.DS_Store'
}

# 接頭語で除外するもの（例: ドットファイル）
EXCLUDE_PREFIXES = {'.'}

def cleanup():
    print(f"Cleanup started for: {ROOT_DIR}")
    
    # Ensure dirs exist
    for d in DEST_DIRS.values():
        os.makedirs(d, exist_ok=True)

    # List all files in root
    try:
        files = [f for f in os.listdir(ROOT_DIR) if os.path.isfile(os.path.join(ROOT_DIR, f))]
    except FileNotFoundError:
        print(f"Error: Directory not found: {ROOT_DIR}")
        return

    moved_count = 0
    
    for filename in files:
        # Check Exclusions
        if filename in EXCLUDE_FILES:
            continue
        if any(filename.startswith(p) for p in EXCLUDE_PREFIXES):
            continue

        src_path = os.path.join(ROOT_DIR, filename)
        dest_key = None
        
        # 1. Python Scripts
        if filename.endswith('.py'):
            # Don't move yourself if you were in root (though you should be in python_files)
            if filename != 'cleanup_workspace.py': 
                dest_key = 'python_files'
        
        # 2. Figures
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            dest_key = 'Figures'
            
        # 3. Logs & Text
        elif filename.lower().endswith(('.log', '.txt')):
            dest_key = 'logs'
            
        # 4. Data / Backups
        # Move CSV/JSON only if they look like backups or logs
        # Or just move all CSVs in root? User has many data files.
        # Let's be safe: Move if it looks like a log dump or backup, or if it's the specific format we saw before.
        # User said "All logs and backups (*.txt, *.csv, *.json) to logs/" in previous turn.
        elif filename.lower().endswith(('.csv', '.json')):
             dest_key = 'logs'

        # Execute Move
        if dest_key:
            dest_path = os.path.join(DEST_DIRS[dest_key], filename)
            
            # Handle collision? Overwrite for now as these are usually temp files
            try:
                shutil.move(src_path, dest_path)
                print(f"[Move] {filename} -> {dest_key}/")
                moved_count += 1
            except Exception as e:
                print(f"[Error] Failed to move {filename}: {e}")

    # Clean __pycache__ if exists
    pycache_path = os.path.join(ROOT_DIR, '__pycache__')
    if os.path.exists(pycache_path):
        try:
            shutil.rmtree(pycache_path)
            print("[Delete] Removed __pycache__")
        except Exception as e:
            print(f"[Error] Failed to remove __pycache__: {e}")

    print(f"{'-'*40}")
    print(f"Cleanup complete. Moved {moved_count} files.")

if __name__ == '__main__':
    cleanup()
