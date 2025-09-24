# Multi-instrument support for NotaGen
# This file defines instrument mappings and multi-instrument combinations

# MIDI program numbers for common instruments
INSTRUMENT_MIDI_PROGRAMS = {
    # Keyboard instruments
    "Piano": 0,
    "Harpsichord": 6,
    "Organ": 16,
    
    # String instruments
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Double Bass": 43,
    "Harp": 46,
    
    # Woodwind instruments
    "Flute": 73,
    "Piccolo": 72,
    "Oboe": 68,
    "Clarinet": 71,
    "Bassoon": 70,
    
    # Brass instruments
    "Trumpet": 56,
    "Horn": 60,
    "Trombone": 57,
    "Tuba": 58,
    
    # Voice
    "Voice": 52,
    "Soprano": 52,
    "Alto": 52,
    "Tenor": 52,
    "Bass": 52,
    "Choir": 52
}

# Clef assignments for instruments
INSTRUMENT_CLEFS = {
    "Piano": ["treble", "bass"],
    "Harpsichord": ["treble", "bass"], 
    "Organ": ["treble", "bass"],
    "Violin": ["treble"],
    "Viola": ["alto"],
    "Cello": ["bass"],
    "Double Bass": ["bass"],
    "Harp": ["treble", "bass"],
    "Flute": ["treble"],
    "Piccolo": ["treble"],
    "Oboe": ["treble"],
    "Clarinet": ["treble"],
    "Bassoon": ["bass"],
    "Trumpet": ["treble"],
    "Horn": ["treble"],
    "Trombone": ["bass"],
    "Tuba": ["bass"],
    "Voice": ["treble"],
    "Soprano": ["treble"],
    "Alto": ["treble"],
    "Tenor": ["treble_8"],
    "Bass": ["bass"],
    "Choir": ["treble"]
}

# Common multi-instrument combinations
MULTI_INSTRUMENT_PRESETS = {
    # Chamber music
    "String Quartet": ["Violin", "Violin", "Viola", "Cello"],
    "Piano Trio": ["Piano", "Violin", "Cello"],
    "String Quintet": ["Violin", "Violin", "Viola", "Cello", "Double Bass"],
    "Wind Quintet": ["Flute", "Oboe", "Clarinet", "Bassoon", "Horn"],
    "Brass Quintet": ["Trumpet", "Trumpet", "Horn", "Trombone", "Tuba"],
    
    # Voice combinations
    "Piano Song": ["Piano", "Voice"],
    "SATB Choir": ["Soprano", "Alto", "Tenor", "Bass"],
    
    # Solo instruments with accompaniment
    "Violin Sonata": ["Violin", "Piano"],
    "Cello Sonata": ["Cello", "Piano"],
    "Flute Sonata": ["Flute", "Piano"]
}

# Extended instrument list including the presets
EXTENDED_INSTRUMENTS = list(INSTRUMENT_MIDI_PROGRAMS.keys()) + list(MULTI_INSTRUMENT_PRESETS.keys())

def parse_instrument_specification(instrument_spec):
    """
    Parse an instrument specification string into a list of instruments.
    
    Examples:
    - "Piano" -> ["Piano"]
    - "Piano+Violin" -> ["Piano", "Violin"]
    - "String Quartet" -> ["Violin", "Violin", "Viola", "Cello"]
    """
    if instrument_spec in MULTI_INSTRUMENT_PRESETS:
        return MULTI_INSTRUMENT_PRESETS[instrument_spec]
    elif '+' in instrument_spec:
        return [inst.strip() for inst in instrument_spec.split('+')]
    else:
        return [instrument_spec]

def generate_abc_voice_section(instruments, voice_prefix="V"):
    """
    Generate the voice definition section for ABC notation.
    
    Args:
        instruments: List of instrument names
        voice_prefix: Prefix for voice names (default "V")
    
    Returns:
        String containing the voice definitions
    """
    voice_lines = []
    
    # Generate voice names and definitions
    voice_names = []
    voice_counter = 1
    
    for instrument in instruments:
        clefs = INSTRUMENT_CLEFS.get(instrument, ["treble"])
        midi_program = INSTRUMENT_MIDI_PROGRAMS.get(instrument, 0)
        
        # Handle keyboard instruments with multiple staves
        if len(clefs) > 1 and instrument in ["Piano", "Harpsichord", "Organ", "Harp"]:
            # For keyboard instruments, create two voices (right and left hand)
            voice_id1 = f"{voice_prefix}{voice_counter}"
            voice_id2 = f"{voice_prefix}{voice_counter+1}"
            voice_names.extend([voice_id1, voice_id2])
            
            voice_lines.append(f'V:{voice_id1} clef={clefs[0]} name="{instrument} Right Hand"')
            voice_lines.append(f'%%MIDI program {midi_program}')
            voice_lines.append(f'V:{voice_id2} clef={clefs[1]} name="{instrument} Left Hand"')
            voice_lines.append(f'%%MIDI program {midi_program}')
            
            voice_counter += 2
        else:
            voice_id = f"{voice_prefix}{voice_counter}"
            voice_names.append(voice_id)
            
            voice_lines.append(f'V:{voice_id} clef={clefs[0]} name="{instrument}"')
            voice_lines.append(f'%%MIDI program {midi_program}')
            
            voice_counter += 1
    
    # Generate score directive
    score_line = f"%%score ({' '.join(voice_names)})"
    
    return score_line + '\n' + '\n'.join(voice_lines)

def is_multi_instrument_prompt(instrumentation):
    """Check if the instrumentation is a multi-instrument specification."""
    return ('+' in instrumentation or 
            instrumentation in MULTI_INSTRUMENT_PRESETS or
            instrumentation in EXTENDED_INSTRUMENTS)

def post_process_multi_instrument(abc_lines, instruments):
    """
    Post-process generated ABC to inject proper multi-instrument voice structure.
    
    Args:
        abc_lines: List of ABC notation lines
        instruments: List of instrument names
        
    Returns:
        Modified ABC lines with proper voice structure
    """
    # Find the insertion point (after title, meter, key, but before first voice/musical content)
    insert_index = 0
    for i, line in enumerate(abc_lines):
        line_stripped = line.strip()
        if line_stripped.startswith(('T:', 'M:', 'L:', 'K:')):
            insert_index = i + 1
        elif line_stripped.startswith(('[V:', '[r:', 'V:')):
            break
        elif line_stripped and not line_stripped.startswith(('T:', 'M:', 'L:', 'K:', '%')):
            # Found musical content without voice marker
            break
    
    # Generate voice section for the instruments
    voice_section = generate_abc_voice_section(instruments)
    voice_lines = [line + '\n' for line in voice_section.split('\n') if line.strip()]
    
    # Insert voice definitions
    result_lines = abc_lines[:insert_index] + voice_lines + abc_lines[insert_index:]
    
    # If the original music doesn't have voice markers, we need to add them
    # This is a simplified approach - in a more sophisticated implementation,
    # we would distribute the musical content across multiple voices
    has_voice_markers = any(line.strip().startswith(('[V:', 'V:')) for line in abc_lines[insert_index:])
    
    if not has_voice_markers:
        # Add a simple V:V1 marker before the first musical content
        music_start_index = insert_index + len(voice_lines)
        for i in range(music_start_index, len(result_lines)):
            line = result_lines[i].strip()
            if line and not line.startswith(('T:', 'M:', 'L:', 'K:', '%', '%%')):
                result_lines.insert(i, 'V:V1\n')
                break
    
    return result_lines

# Test the functions
if __name__ == "__main__":
    # Test parsing
    test_specs = ["Piano", "Piano+Violin", "String Quartet", "Piano+Violin+Cello"]
    
    for spec in test_specs:
        instruments = parse_instrument_specification(spec)
        print(f"'{spec}' -> {instruments}")
        print("Voice section:")
        print(generate_abc_voice_section(instruments))
        print("-" * 50)