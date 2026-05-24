import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path.cwd() / "src"))

from mufinder.services.project_service import ProjectService

projects = [
    "/Users/bhuddart/Library/CloudStorage/OneDrive-Nexus365/Oxford/Research/MuFinder/Cu/Cu_diffusion.mufinder",
    "/Users/bhuddart/Library/CloudStorage/OneDrive-Nexus365/Oxford/Research/Projects/Current/MnTe/HubbardU/MnTe_mufinder.mufinder"
]

for project_path in projects:
    print(f"--- Loading project: {project_path} ---")
    try:
        service = ProjectService()
        project = service.load_project(Path(project_path))
        print(f"Successfully loaded {project_path} in ProjectService.load_project")
    except Exception as e:
        print(f"Failed in ProjectService.load_project for {project_path}")
        import traceback
        traceback.print_exc()
    print("\n")
