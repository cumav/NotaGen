# NotaGen Standalone Inference

This document explains how to run NotaGen inference without using the Gradio interface, using plain Python code.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Check Available Combinations

To see all available composer and instrument combinations:

```bash
python standalone_inference.py --list-combinations
```

To find combinations for a specific composer:

```bash
python standalone_inference.py --list-combinations | grep -i debussy
```

### 3. Generate Music for Debussy + Piano

```bash
python standalone_inference.py --composer "Debussy" --instrument "Keyboard"
```

This will generate music in the style of Claude Debussy for keyboard/piano.

## Model Weights

**Important**: You need to download the model weights file before running inference. The script expects a file named `weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth` in the `gradio/` directory.

If you have the weights file in a different location, you can specify it:

```bash
python standalone_inference.py --composer "Debussy" --instrument "Keyboard" --weights /path/to/your/weights.pth
```

## Usage Examples

### Basic Usage
```bash
# Generate Debussy piano music
python standalone_inference.py --composer "Debussy" --instrument "Keyboard"

# Generate Bach chamber music
python standalone_inference.py --composer "Bach" --instrument "Chamber"

# Generate Mozart orchestral music
python standalone_inference.py --composer "Mozart" --instrument "Orchestral"
```

### Advanced Options
```bash
# Specify output file
python standalone_inference.py --composer "Debussy" --instrument "Keyboard" --output my_composition.abc

# Specify period explicitly
python standalone_inference.py --period "Romantic" --composer "Debussy" --instrument "Keyboard"

# Use custom weights file
python standalone_inference.py --composer "Debussy" --instrument "Keyboard" --weights /path/to/weights.pth
```

## Available Combinations

The script supports 112 different combinations of period, composer, and instrument. Some popular examples:

### Baroque Period
- Bach, Johann Sebastian + Keyboard/Chamber/Choral/Orchestral
- Handel, George Frideric + Chamber/Keyboard/Orchestral
- Vivaldi, Antonio + Chamber/Orchestral

### Classical Period  
- Beethoven, Ludwig van + Keyboard/Chamber/Orchestral
- Mozart, Wolfgang Amadeus + Keyboard/Chamber/Orchestral/Choral
- Haydn, Joseph + Keyboard/Chamber/Orchestral

### Romantic Period
- **Debussy, Claude + Keyboard/Art Song** ← Your requested combination!
- Chopin, Frederic + Keyboard/Art Song
- Brahms, Johannes + Keyboard/Chamber/Orchestral/Choral
- Liszt, Franz + Keyboard
- Rachmaninoff, Sergei + Keyboard/Choral

## Output Format

The script generates music in ABC notation format, which is a text-based musical notation system. The output includes:

1. **Console output**: Real-time generation progress and the final ABC notation
2. **File output**: Automatically saved with timestamp and composer info, or specify with `--output`

Example output filename: `generated_20240924_143022_Romantic_Debussy_Claude_Keyboard.abc`

## Converting ABC to Other Formats

The generated ABC files can be converted to other formats:

- **MIDI**: Use `abc2midi` tool
- **MusicXML**: Use the included `abc2xml.py` script  
- **Sheet music**: Use `abcm2ps` or similar tools
- **Audio**: Convert MIDI to audio using tools like FluidSynth

## Troubleshooting

### "Model weights file not found"
Download the model weights file and place it in the correct location, or use `--weights` to specify the path.

### "No matching combinations found"
Use `--list-combinations` to see all available options. Note that composer names must match exactly (e.g., "Debussy, Claude" not just "Claude Debussy").

### Memory issues
The model requires significant memory. If you get out-of-memory errors:
- Use CPU instead of GPU (automatic fallback)
- Close other applications
- Consider using a machine with more RAM

### Generation takes too long
The generation process can take several minutes. The script has a 10-minute timeout. If it's taking too long:
- Try a different combination
- Ensure you're using the correct weights file
- Check that all dependencies are properly installed

## Script Options

```
usage: standalone_inference.py [-h] [--composer COMPOSER] [--instrument INSTRUMENT] 
                               [--period PERIOD] [--list-combinations] [--output OUTPUT] 
                               [--weights WEIGHTS]

options:
  -h, --help            show this help message and exit
  --composer COMPOSER   Composer name (e.g., "Debussy" or "Debussy, Claude")
  --instrument INSTRUMENT
                        Instrument type (e.g., "Keyboard", "Piano")
  --period PERIOD       Musical period (e.g., "Romantic", "Classical", "Baroque")  
  --list-combinations   List all available combinations
  --output OUTPUT       Output file path for generated ABC notation
  --weights WEIGHTS     Path to model weights file (overrides config.py)
```