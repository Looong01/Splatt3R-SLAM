#!/usr/bin/env python3
"""
Diagnostic tool for Splatt3R-SLAM installation issues.
Run this script to identify which dependencies are missing or causing import errors.
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"

def main():
    print("="*70)
    print("Splatt3R-SLAM Dependency Diagnostic Tool")
    print("="*70)
    print()
    
    # Core Python packages
    print("Testing core dependencies...")
    print("-" * 70)
    core_deps = [
        ('torch', 'PyTorch'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
    ]
    
    missing_core = []
    for module, package in core_deps:
        success, error = test_import(module)
        status = "✓" if success else "✗"
        print(f"{status} {package:20s} ({module})")
        if not success:
            missing_core.append((package, error))
            if error:
                print(f"  Error: {error}")
    
    print()
    print("Testing Splatt3R-specific dependencies...")
    print("-" * 70)
    splatt3r_deps = [
        ('lietorch', 'lietorch'),
        ('einops', 'einops'),
        ('lightning', 'lightning'),
        ('lpips', 'lpips'),
        ('omegaconf', 'omegaconf'),
        ('huggingface_hub', 'huggingface_hub'),
        ('git', 'gitpython'),
    ]
    
    missing_splatt3r = []
    for module, package in splatt3r_deps:
        success, error = test_import(module)
        status = "✓" if success else "✗"
        print(f"{status} {package:20s} ({module})")
        if not success:
            missing_splatt3r.append((package, error))
            if error:
                print(f"  Error: {error}")
    
    print()
    print("Testing Splatt3R-SLAM modules...")
    print("-" * 70)
    slam_modules = [
        'splatt3r_slam.config',
        'splatt3r_slam.frame',
        'splatt3r_slam.global_opt',
        'splatt3r_slam.dataloader',
        'splatt3r_slam.splatt3r_utils',
    ]
    
    failed_modules = []
    for module in slam_modules:
        success, error = test_import(module)
        status = "✓" if success else "✗"
        print(f"{status} {module}")
        if not success:
            failed_modules.append((module, error))
            if error:
                print(f"  Error: {error}")
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    all_missing = missing_core + missing_splatt3r
    
    if not all_missing and not failed_modules:
        print("✓ All dependencies are installed correctly!")
        print()
        print("You should be able to run:")
        print("  python main_splatt3r.py --dataset <path> --config config/base.yaml")
    else:
        print("✗ Issues found:")
        print()
        
        if missing_core:
            print("Missing core dependencies:")
            for package, _ in missing_core:
                print(f"  - {package}")
        
        if missing_splatt3r:
            print()
            print("Missing Splatt3R-specific dependencies:")
            for package, _ in missing_splatt3r:
                print(f"  - {package}")
        
        if failed_modules:
            print()
            print("Failed module imports:")
            for module, error in failed_modules:
                print(f"  - {module}")
                if "No module named" in error:
                    # Extract the missing module name
                    import re
                    match = re.search(r"No module named '(\w+)'", error)
                    if match:
                        print(f"    Missing: {match.group(1)}")
        
        print()
        print("Installation commands:")
        print("-" * 70)
        
        # Collect unique missing packages
        packages_to_install = []
        special_installs = []
        
        for package, _ in all_missing:
            if package == 'lietorch':
                special_installs.append("pip install git+https://github.com/princeton-vl/lietorch.git")
            else:
                packages_to_install.append(package)
        
        if special_installs:
            print()
            print("# Install lietorch first (required):")
            for cmd in special_installs:
                print(cmd)
        
        if packages_to_install:
            print()
            print("# Install other dependencies:")
            print(f"pip install {' '.join(packages_to_install)}")
        
        print()
        print("For complete installation instructions, see:")
        print("  - README.md")
        print("  - QUICKSTART.md")
        print("  - TROUBLESHOOTING.md")
    
    print("="*70)
    
    # Return exit code
    return 0 if (not all_missing and not failed_modules) else 1

if __name__ == "__main__":
    sys.exit(main())
