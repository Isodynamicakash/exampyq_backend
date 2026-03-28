#!/usr/bin/env python3
"""
test_mathpix.py
===============
Test script for mathpix.py service.

Usage:
    python test_mathpix.py <path_to_pdf>

Example:
    python test_mathpix.py sample.pdf
    python test_mathpix.py /home/user/documents/exam_paper.pdf

Requirements:
    - Set MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables
    - Install: aiohttp, aiofiles
"""
# Load .env file first!
from dotenv import load_dotenv
load_dotenv()
import asyncio
import sys
import os
from pathlib import Path
import uuid

# Import the mathpix service
# Make sure mathpix.py is in the same directory or in your Python path
try:
    import mathpix
except ImportError:
    print("❌ Error: Cannot import mathpix.py")
    print("Make sure mathpix.py is in the same directory or in your Python path")
    sys.exit(1)


async def test_mathpix(pdf_path: str):
    """Test the MathPix pipeline with a given PDF file."""
    
    # Validate PDF path
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return False
    
    if not pdf_file.suffix.lower() == '.pdf':
        print(f"⚠️  Warning: File doesn't have .pdf extension: {pdf_path}")
    
    # Check environment variables
    if not os.getenv("MATHPIX_APP_ID"):
        print("❌ Error: MATHPIX_APP_ID environment variable not set")
        print("Set it with: export MATHPIX_APP_ID='your_app_id'")
        return False
    
    if not os.getenv("MATHPIX_APP_KEY"):
        print("❌ Error: MATHPIX_APP_KEY environment variable not set")
        print("Set it with: export MATHPIX_APP_KEY='your_app_key'")
        return False
    
    print("="*70)
    print("🚀 MathPix Test Starting")
    print("="*70)
    print(f"📄 PDF File: {pdf_file.name}")
    print(f"📏 File Size: {pdf_file.stat().st_size / 1024:.2f} KB")
    print()
    
    # Generate unique job ID
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    print(f"🆔 Job ID: {job_id}")
    print()
    
    try:
        # Read PDF bytes
        print("📖 Reading PDF file...")
        with open(pdf_file, 'rb') as f:
            pdf_bytes = f.read()
        print(f"✓ Read {len(pdf_bytes)} bytes")
        print()
        
        # Run MathPix pipeline
        print("🔄 Starting MathPix pipeline...")
        print("   This may take 30-120 seconds depending on PDF size...")
        print()
        
        job_dir = await mathpix.run_mathpix_pipeline(
            pdf_bytes=pdf_bytes,
            filename=pdf_file.name,
            job_id=job_id
        )
        
        print("✅ MathPix pipeline completed successfully!")
        print()
        
        # Display results
        print("="*70)
        print("📦 RESULTS")
        print("="*70)
        
        # Check output directory
        print(f"📁 Job Directory: {job_dir}")
        print()
        
        # Check LaTeX file
        tex_file = job_dir / "output.tex"
        if tex_file.exists():
            tex_size = tex_file.stat().st_size
            print(f"✅ LaTeX File: output.tex ({tex_size:,} bytes)")
            
            # Show first 500 characters of LaTeX
            tex_content = tex_file.read_text(encoding='utf-8', errors='ignore')
            print()
            print("📝 LaTeX Preview (first 500 chars):")
            print("-" * 70)
            print(tex_content[:500])
            if len(tex_content) > 500:
                print("... (truncated)")
            print("-" * 70)
        else:
            print("❌ LaTeX file not found!")
        
        print()
        
        # Check images
        images_dir = job_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*"))
            print(f"🖼️  Images: {len(image_files)} files extracted")
            
            if image_files:
                print()
                print("   Image files:")
                for img in sorted(image_files)[:10]:  # Show first 10
                    img_size = img.stat().st_size / 1024  # KB
                    print(f"   • {img.name} ({img_size:.1f} KB)")
                
                if len(image_files) > 10:
                    print(f"   ... and {len(image_files) - 10} more images")
        else:
            print("⚠️  No images directory found")
        
        print()
        print("="*70)
        print("📍 Full Output Location:")
        print(f"   {job_dir.absolute()}")
        print("="*70)
        print()
        
        # Provide next steps
        print("💡 Next Steps:")
        print(f"   1. View LaTeX: cat {job_dir / 'output.tex'}")
        print(f"   2. View Images: ls -lh {job_dir / 'images'}")
        print(f"   3. Parse LaTeX: Use your parser on {job_dir / 'output.tex'}")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("="*70)
        print("❌ ERROR")
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        
        # Debug info
        import traceback
        print("Full traceback:")
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)
        
        return False


def main():
    """Main entry point."""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_mathpix.py <path_to_pdf>")
        print()
        print("Example:")
        print("  python test_mathpix.py sample.pdf")
        print("  python test_mathpix.py /home/user/exam_paper.pdf")
        print()
        print("Environment Variables Required:")
        print("  MATHPIX_APP_ID  - Your MathPix App ID")
        print("  MATHPIX_APP_KEY - Your MathPix App Key")
        print()
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Run async test
    success = asyncio.run(test_mathpix(pdf_path))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()