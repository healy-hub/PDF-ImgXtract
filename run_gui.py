# -*- coding: utf-8 -*-
import sys
import os

def run():
    """
    Sets up the Python path and runs the main GUI application.
    This script acts as a simple entry point for the application,
    making it easier to run and package.
    """
    # Add the project root directory to the Python path.
    # This is the directory that contains the 'pdf_extractor' package.
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # We add the root, not the 'src' or package dir, because the imports
    # are written as 'from pdf_extractor...'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from pdf_extractor.main import main
    except ImportError as e:
        print("Error: Could not import the main application.")
        print(f"Please ensure the directory structure is correct and you are running from the 'PDF2PNG' root folder.")
        print(f"Import Error: {e}")
        sys.exit(1)

    # Call the main function from the application package
    main()

if __name__ == "__main__":
    run()
