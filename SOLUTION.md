# Solution: Running NotaGen Inference Without Gradio

## The Problem
You wanted to know how to run inference without using Gradio - just plain code to select Debussy as composer and piano as instrument.

## The Solution

### Option 1: Direct Function Call (Simple)

```python
# Import the inference function
from inference import inference_patch

# Select Debussy + Piano (the exact parameters from the dataset)
period = "Romantic"
composer = "Debussy, Claude"  
instrument = "Keyboard"  # This is how piano is labeled in the dataset

# Run inference
result = inference_patch(period, composer, instrument)

# result will contain the generated ABC notation music
print(result)
```

### Option 2: Standalone Script (Recommended)

I created a standalone script that handles everything for you:

```bash
# Generate Debussy piano music
python standalone_inference.py --composer "Debussy" --instrument "Piano"

# Or more specifically
python standalone_inference.py --period "Romantic" --composer "Debussy, Claude" --instrument "Keyboard"
```

The script automatically:
- Finds the correct combination (maps "Piano" to "Keyboard")
- Loads the model weights
- Runs inference  
- Saves the output to a file

### Option 3: Interactive Demo

Run the demo to see exactly what parameters to use:

```bash
python simple_demo.py
```

This shows you the exact parameters and code needed.

## Key Information

**Correct Parameters for Debussy + Piano:**
- Period: `"Romantic"`
- Composer: `"Debussy, Claude"`
- Instrument: `"Keyboard"` (not "Piano")

**Available Debussy Combinations:**
1. `"Romantic", "Debussy, Claude", "Art Song"`
2. `"Romantic", "Debussy, Claude", "Keyboard"` ← This is what you want

## Requirements

1. **Dependencies**: Install with `pip install -r requirements.txt` and `pip install torch`
2. **Model Weights**: You need the weights file `weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth`

## Files Created

1. **`standalone_inference.py`** - Main script for running inference without Gradio
2. **`simple_demo.py`** - Shows exactly what parameters to use
3. **`README_standalone.md`** - Complete documentation
4. **`SOLUTION.md`** - This summary

## Quick Test (Without Model Weights)

To verify the combination works:

```bash
# See all available combinations
python standalone_inference.py --list-combinations | grep Debussy

# Test the parameter matching
python standalone_inference.py --composer "Debussy" --instrument "Piano"
```

You'll see it correctly identifies "Romantic, Debussy, Claude, Keyboard" even when you search for "Piano".

## Next Steps

1. Download the model weights file
2. Place it in the `gradio/` directory or specify with `--weights`
3. Run: `python standalone_inference.py --composer "Debussy" --instrument "Piano"`
4. The generated music will be saved as an ABC notation file

That's it! You now have multiple ways to run NotaGen inference without Gradio, specifically set up for Debussy + piano.