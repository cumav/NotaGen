#!/usr/bin/env python3
"""
Simple Demo: How to run NotaGen inference without Gradio
========================================================

This is a minimal example showing exactly how to select Debussy as composer 
and piano as instrument, and call the inference function with plain code.

This demonstrates the parameters and setup needed without running the full model.
"""

import sys
from pathlib import Path

# Add gradio directory to path
script_dir = Path(__file__).parent.resolve()
gradio_dir = script_dir / "gradio"
sys.path.insert(0, str(gradio_dir))

def demo_debussy_piano():
    """
    Demo showing exactly how to select Debussy + Piano for inference
    """
    
    print("NotaGen Inference Demo - Debussy + Piano")
    print("=" * 50)
    
    # Step 1: Define the parameters for Debussy + Piano
    # Based on the prompts.txt file, the correct combination is:
    period = "Romantic"
    composer = "Debussy, Claude" 
    instrument = "Keyboard"  # This is the actual instrument name in the data
    
    print(f"Selected parameters:")
    print(f"  Period: {period}")
    print(f"  Composer: {composer}")
    print(f"  Instrument: {instrument}")
    print()
    
    # Step 2: Show what the prompt lines would look like
    prompt_lines = [
        '%' + period + '\n',
        '%' + composer + '\n', 
        '%' + instrument + '\n'
    ]
    
    print("Prompt lines that would be sent to the model:")
    for i, line in enumerate(prompt_lines):
        print(f"  {i+1}. {repr(line)}")
    print()
    
    # Step 3: Show how you would call the inference function
    print("Code to run inference:")
    print("-" * 30)
    print("# Import the inference function")
    print("from inference import inference_patch")
    print()
    print("# Call inference with Debussy + Piano parameters")
    print(f'result = inference_patch("{period}", "{composer}", "{instrument}")')
    print()
    print("# The result will be ABC notation music generated in Debussy's style for piano")
    print()
    
    # Step 4: Show standalone script usage
    print("Or use the standalone script:")
    print("-" * 30)
    print("python standalone_inference.py --composer 'Debussy' --instrument 'Piano'")
    print("python standalone_inference.py --composer 'Claude' --instrument 'Keyboard'")
    print("python standalone_inference.py --period 'Romantic' --composer 'Debussy, Claude' --instrument 'Keyboard'")
    print()
    
    # Step 5: Show all Debussy combinations available
    print("All available Debussy combinations:")
    print("-" * 30)
    debussy_combinations = [
        ("Romantic", "Debussy, Claude", "Art Song"),
        ("Romantic", "Debussy, Claude", "Keyboard"),
    ]
    
    for period, composer, instrument in debussy_combinations:
        print(f"  - Period: {period}, Composer: {composer}, Instrument: {instrument}")
    
    print()
    print("Note: You need the model weights file to actually run inference.")
    print("The script will guide you on where to place or specify the weights file.")


def show_all_keyboard_composers():
    """Show all composers available for keyboard/piano"""
    
    print("\nAll composers available for Keyboard/Piano:")
    print("=" * 50)
    
    # Load combinations from prompts file
    prompts_file = gradio_dir / "prompts.txt"
    
    if prompts_file.exists():
        keyboard_composers = []
        with open(prompts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove line numbers
                    if '.' in line:
                        line = line.split('.', 1)[1]
                    
                    parts = line.split('_')
                    if len(parts) == 3:
                        period, composer, instrument = parts
                        if 'Keyboard' in instrument:
                            keyboard_composers.append((period, composer, instrument))
        
        for period, composer, instrument in sorted(keyboard_composers):
            print(f"  {period:10} | {composer:25} | {instrument}")
    else:
        print("Could not find prompts.txt file")


if __name__ == "__main__":
    demo_debussy_piano()
    show_all_keyboard_composers()