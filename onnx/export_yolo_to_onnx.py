"""
Export YOLO11n model to ONNX format for ONNX Runtime inference.
This script must be run manually to generate yolo11n.onnx.

Usage:
    python onnx/export_yolo_to_onnx.py

Requirements:
    - ultralytics package installed
    - yolo11n.pt model file present

Output:
    - yolo11n.onnx in the root directory
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def export_yolo_to_onnx(yolo_model_path="yolo11n.pt", output_path="yolo11n.onnx"):
    """
    Export YOLO model to ONNX format with dynamic batch size.
    
    Args:
        yolo_model_path: Path to YOLO PyTorch model (.pt file)
        output_path: Path to output ONNX model (.onnx file)
    """
    # Check if input model exists
    if not os.path.exists(yolo_model_path):
        print(f"❌ Error: YOLO model not found at {yolo_model_path}")
        print("   Please ensure yolo11n.pt is in the root directory")
        return False
    
    print(f"📦 Loading YOLO model from {yolo_model_path}...")
    try:
        model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return False
    
    print(f"🔄 Exporting to ONNX format...")
    print(f"   Output: {output_path}")
    print(f"   Dynamic batch size: enabled")
    print(f"   Opset: 12 (compatible with ONNX Runtime)")
    
    try:
        # Export to ONNX
        # dynamic=True enables dynamic batch size
        # simplify=True optimizes the graph
        # opset=12 ensures compatibility with ONNX Runtime
        success = model.export(
            format="onnx",
            imgsz=640,  # Input image size (YOLO11n default)
            dynamic=True,  # Enable dynamic batch size
            simplify=True,  # Optimize graph
            opset=12  # ONNX opset version (compatible with ONNX Runtime)
        )
        
        # Ultralytics exports to same directory with .onnx extension
        # Rename to desired output path
        exported_path = yolo_model_path.replace('.pt', '.onnx')
        if os.path.exists(exported_path):
            if exported_path != output_path:
                import shutil
                shutil.move(exported_path, output_path)
                print(f"✓ Model exported and moved to {output_path}")
            else:
                print(f"✓ Model exported to {output_path}")
            
            # Verify file exists
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"✓ ONNX model file size: {file_size:.2f} MB")
                print(f"✓ Export successful!")
                return True
            else:
                print(f"❌ Error: Exported file not found at {output_path}")
                return False
        else:
            print(f"❌ Error: Export failed - file not found at {exported_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        print(f"   Make sure ultralytics package is up to date")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO to ONNX Export Script")
    print("=" * 60)
    print()
    
    # Export YOLO model
    success = export_yolo_to_onnx()
    
    if success:
        print()
        print("=" * 60)
        print("✅ Export completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Install ONNX Runtime: pip install onnxruntime")
        print("2. Set USE_ONNX_YOLO = True in yolo3d_detector.py")
        print("3. Run your application to test ONNX inference")
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("❌ Export failed!")
        print("=" * 60)
        sys.exit(1)
