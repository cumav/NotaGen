#!/usr/bin/env python3
"""
Minimal Example: Direct inference call for Debussy + Piano

This is the simplest possible code to run NotaGen inference 
for Debussy piano music without Gradio.

Prerequisites:
1. Model weights file must be available
2. All dependencies must be installed
"""

def main():
    # Change to gradio directory and import inference
    import sys
    import os
    from pathlib import Path
    
    # Add gradio directory to Python path
    script_dir = Path(__file__).parent.resolve()
    gradio_dir = script_dir / "gradio"
    sys.path.insert(0, str(gradio_dir))
    
    # Change to gradio directory (needed for relative imports)
    original_dir = os.getcwd()
    os.chdir(gradio_dir)
    
    try:
        # Import the inference function
        # This will automatically load the model when imported
        from inference import inference_patch
        
        # Define parameters for Debussy + Piano
        period = "Romantic"
        composer = "Debussy, Claude"
        instrument = "Keyboard"  # Piano is called "Keyboard" in the dataset
        
        print(f"Generating music for {composer} ({period}) - {instrument}")
        print("This may take several minutes...")
        print("=" * 50)
        
        # Run inference - this is the core line you requested
        result = inference_patch(period, composer, instrument)
        
        if result:
            print("\nGeneration completed!")
            print("Generated ABC notation:")
            print("-" * 30)
            print(result)
            
            # Save to file
            filename = "debussy_piano_generated.abc"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\nSaved to: {filename}")
        else:
            print("Generation failed or returned empty result")
            
    except FileNotFoundError as e:
        if "weights" in str(e).lower():
            print("Error: Model weights file not found!")
            print("Please ensure you have the weights file:")
            print("  weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth")
            print("Place it in the gradio/ directory")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()