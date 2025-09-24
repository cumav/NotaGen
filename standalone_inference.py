#!/usr/bin/env python3
"""
Standalone NotaGen Inference Script
===================================

This script allows you to run NotaGen inference without using Gradio.
You can generate music by specifying a composer and instrument combination.

Usage:
    python standalone_inference.py --composer "Debussy, Claude" --instrument "Keyboard" --period "Romantic"

Requirements:
    - The model weights file specified in config.py must be available
    - All dependencies from requirements.txt must be installed
"""

import os
import sys
import argparse
import time
import torch
from pathlib import Path

# Add gradio directory to path to import the inference modules
script_dir = Path(__file__).parent.resolve()
gradio_dir = script_dir / "gradio"
sys.path.insert(0, str(gradio_dir))

try:
    from utils import *
    from config import *
    from transformers import GPT2Config
    from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
    from abctoolkit.transpose import Note_list, Pitch_sign_list
    from abctoolkit.duration import calculate_bartext_duration
    
    # We'll import the inference function later to avoid loading model during import
    inference_patch = None
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you are running this script from the repository root and have installed all dependencies.")
    sys.exit(1)


def load_valid_combinations(prompts_file):
    """Load valid combinations of period, composer, and instrument from prompts.txt"""
    valid_combinations = set()
    
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found!")
        return valid_combinations
    
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove line numbers (e.g., "1.Baroque_Bach..." -> "Baroque_Bach...")
                if '.' in line:
                    line = line.split('.', 1)[1]
                
                parts = line.split('_')
                if len(parts) == 3:
                    period, composer, instrument = parts
                    valid_combinations.add((period, composer, instrument))
    
    return valid_combinations


def load_model(weights_path=None):
    """Load and initialize the NotaGen model"""
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Initialize patchilizer
    patchilizer = Patchilizer()
    
    # Create model configuration
    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                              max_length=PATCH_LENGTH,
                              max_position_embeddings=PATCH_LENGTH,
                              n_embd=HIDDEN_SIZE,
                              num_attention_heads=HIDDEN_SIZE // 64,
                              vocab_size=1)
    byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                             max_length=PATCH_SIZE + 1,
                             max_position_embeddings=PATCH_SIZE + 1,
                             hidden_size=HIDDEN_SIZE,
                             num_attention_heads=HIDDEN_SIZE // 64,
                             vocab_size=128)

    # Initialize model
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config).to(device)
    
    # Prepare model for training (without actual training)
    def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
        """Prepare model for k-bit training."""
        model = model.to(dtype=torch.float16)
        
        # Disable gradients for embedding layers
        for name, param in model.named_parameters():
            if 'embed' in name.lower():
                param.requires_grad = False
        
        return model
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    # Load weights
    weights_file = weights_path if weights_path else INFERENCE_WEIGHTS_PATH
    print(f"Loading weights from: {weights_file}")
    
    checkpoint = torch.load(weights_file, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, patchilizer, device


def inference_standalone(period, composer, instrumentation, model=None, patchilizer=None, device=None, weights_path=None):
    """Standalone inference function that doesn't rely on global variables"""
    
    # Load model if not provided
    if model is None or patchilizer is None or device is None:
        model, patchilizer, device = load_model(weights_path)
    
    # Import utility functions from inference module for processing
    import re
    
    # Define helper functions (from original inference.py)
    def complete_brackets(s):
        stack = []
        bracket_map = {'{': '}', '[': ']', '(': ')'}
        
        for char in s:
            if char in bracket_map:
                stack.append(bracket_map[char])
            elif char in bracket_map.values():
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
        
        return s + ''.join(reversed(stack))
    
    def rest_unreduce(abc_lines):
        """Convert reduced ABC notation to unreduced format"""
        
        # Extract metadata and tunebody
        metadata_lines = []
        tunebody_lines = []
        metadata_flag = True
        
        for line in abc_lines:
            if line.startswith('[r:'):
                metadata_flag = False
            
            if metadata_flag and not line.startswith('[r:'):
                metadata_lines.append(line)
            else:
                tunebody_lines.append(line)
        
        # Process tunebody to extract voice parts
        part_symbol_list = []
        for line in tunebody_lines:
            pattern = r'\[V:(\d+)\]'
            matches = re.findall(pattern, line)
            for match in matches:
                symbol = f'V:{match}'
                if symbol not in part_symbol_list:
                    part_symbol_list.append(symbol)
        
        z_symbol_list = []
        x_symbol_list = []
        ref_dur = 4  # Default duration
        
        unreduced_tunebody_lines = []
        
        for i, line in enumerate(tunebody_lines):
            unreduced_line = ''
            
            line = re.sub(r'^\[r:[^\]]*\]', '', line)
            
            pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
            matches = re.findall(pattern, line)
            
            line_bar_dict = {}
            for match in matches:
                key = f'V:{match[0]}'
                value = match[1]
                line_bar_dict[key] = value
            
            # Calculate duration and collect barline
            dur_dict = {}  
            for symbol, bartext in line_bar_dict.items():
                # Simple barline extraction
                right_barline = '|'  # Default barline
                bartext_clean = bartext.rstrip('|')
                try:
                    bar_dur = calculate_bartext_duration(bartext_clean)
                except:
                    bar_dur = None
                if bar_dur is not None:
                    if bar_dur not in dur_dict.keys():
                        dur_dict[bar_dur] = 1
                    else:
                        dur_dict[bar_dur] += 1
            
            try:
                ref_dur = max(dur_dict, key=dur_dict.get)
            except:
                pass  # use last ref_dur
            
            if i == 0:
                prefix_left_barline = line.split('[V:')[0] if '[V:' in line else ''
            else:
                prefix_left_barline = ''
            
            for symbol in part_symbol_list:
                if symbol in line_bar_dict.keys():
                    symbol_bartext = line_bar_dict[symbol]
                else:
                    if symbol in z_symbol_list:
                        symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + '|'
                    elif symbol in x_symbol_list:
                        symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + '|'
                    else:
                        symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + '|'
                unreduced_line += '[' + symbol + ']' + symbol_bartext
            
            unreduced_tunebody_lines.append(unreduced_line + '\n')
        
        unreduced_lines = metadata_lines + unreduced_tunebody_lines
        return unreduced_lines
    
    # Main inference logic (adapted from original inference_patch)
    prompt_lines = [
        '%' + period + '\n',
        '%' + composer + '\n',
        '%' + instrumentation + '\n'
    ]
    
    while True:
        failure_flag = False
        
        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]
        
        start_time = time.time()
        
        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        context_tunebody_byte_list = []
        metadata_byte_list = []
        
        print(''.join(byte_list), end='')
        
        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch in prompt_patches]
        prompt_patches.insert(0, bos_patch)
        
        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)
        
        end_flag = False
        cut_index = None
        tunebody_flag = False
        
        with torch.inference_mode():
            while True:
                # Use autocast only for CUDA
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        predicted_patch = model.generate(input_patches.unsqueeze(0),
                                                        top_k=TOP_K,
                                                        top_p=TOP_P,
                                                        temperature=TEMPERATURE)
                else:
                    predicted_patch = model.generate(input_patches.unsqueeze(0),
                                                    top_k=TOP_K,
                                                    top_p=TOP_P,
                                                    temperature=TEMPERATURE)
                
                if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):
                    tunebody_flag = True
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    
                    if device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                            top_k=TOP_K,
                                                            top_p=TOP_P,
                                                            temperature=TEMPERATURE)
                    else:
                        predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                        top_k=TOP_K,
                                                        top_p=TOP_P,
                                                        temperature=TEMPERATURE)
                    
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
                
                if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                    end_flag = True
                    break
                
                next_patch = patchilizer.decode([predicted_patch])
                
                for char in next_patch:
                    byte_list.append(char)
                    if tunebody_flag:
                        context_tunebody_byte_list.append(char)
                    else:
                        metadata_byte_list.append(char)
                    print(char, end='')
                
                patch_end_flag = False
                for j in range(len(predicted_patch)):
                    if patch_end_flag:
                        predicted_patch[j] = patchilizer.special_token_id
                    if predicted_patch[j] == patchilizer.eos_token_id:
                        patch_end_flag = True
                
                predicted_patch = torch.tensor([predicted_patch], device=device)
                input_patches = torch.cat([input_patches, predicted_patch], dim=1)
                
                if len(byte_list) > 102400:
                    failure_flag = True
                    break
                if time.time() - start_time > 10 * 60:
                    failure_flag = True
                    break
                
                if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
                    print('Stream generating...')
                    
                    metadata = ''.join(metadata_byte_list)
                    context_tunebody = ''.join(context_tunebody_byte_list)
                    
                    if '\n' not in context_tunebody:
                        break
                    
                    context_tunebody_lines = context_tunebody.split('\n')
                    if not context_tunebody.endswith('\n'):
                        context_tunebody_lines = [context_tunebody_lines[i] + '\n' for i in range(len(context_tunebody_lines) - 1)] + [context_tunebody_lines[-1]]
                    else:
                        context_tunebody_lines = [context_tunebody_lines[i] + '\n' for i in range(len(context_tunebody_lines))]
                    
                    cut_index = len(context_tunebody_lines) // 2
                    abc_code_slice = metadata + ''.join(context_tunebody_lines[-cut_index:])
                    
                    input_patches = patchilizer.encode_generate(abc_code_slice)
                    
                    input_patches = [item for sublist in input_patches for item in sublist]
                    input_patches = torch.tensor([input_patches], device=device)
                    input_patches = input_patches.reshape(1, -1)
                    
                    context_tunebody_byte_list = list(''.join(context_tunebody_lines[-cut_index:]))
        
        if not failure_flag:
            abc_text = ''.join(byte_list)
            
            # Unreduce
            abc_lines = abc_text.split('\n')
            abc_lines = list(filter(None, abc_lines))
            abc_lines = [line + '\n' for line in abc_lines]
            try:
                unreduced_abc_lines = rest_unreduce(abc_lines)
            except Exception as e:
                print(f"\nWarning: rest_unreduce failed: {e}")
                failure_flag = True
                pass
            else:
                unreduced_abc_lines = [line for line in unreduced_abc_lines if not(line.startswith('%') and not line.startswith('%%'))]
                unreduced_abc_lines = ['X:1\n'] + unreduced_abc_lines
                unreduced_abc_text = ''.join(unreduced_abc_lines)
                return unreduced_abc_text
        
        if failure_flag:
            print("\nGeneration failed, retrying...")
            continue
        else:
            break
    
    return None
    """Find combinations that match the given queries"""
    matches = []
    
    for period, composer, instrument in valid_combinations:
        match = True
        
        if composer_query and composer_query.lower() not in composer.lower():
            match = False
        if instrument_query and instrument_query.lower() not in instrument.lower():
            match = False
        if period_query and period_query.lower() != period.lower():
            match = False
            
        if match:
            matches.append((period, composer, instrument))
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Generate music using NotaGen without Gradio interface")
    parser.add_argument('--composer', type=str, help='Composer name (e.g., "Debussy" or "Debussy, Claude")')
    parser.add_argument('--instrument', type=str, help='Instrument type (e.g., "Keyboard", "Piano")')
    parser.add_argument('--period', type=str, help='Musical period (e.g., "Romantic", "Classical", "Baroque")')
    parser.add_argument('--list-combinations', action='store_true', help='List all available combinations')
    parser.add_argument('--output', type=str, default=None, help='Output file path for generated ABC notation')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights file (overrides config.py)')
    
    args = parser.parse_args()
    
    # Load valid combinations
    prompts_file = gradio_dir / "prompts.txt"
    valid_combinations = load_valid_combinations(prompts_file)
    
    if not valid_combinations:
        print("Error: Could not load valid combinations from prompts.txt")
        return 1
    
    # List combinations if requested
    if args.list_combinations:
        print("Available combinations:")
        print("======================")
        for period, composer, instrument in sorted(valid_combinations):
            print(f"Period: {period}, Composer: {composer}, Instrument: {instrument}")
        return 0
    
    # Find matching combinations
    matches = find_matching_combinations(
        valid_combinations, 
        composer_query=args.composer,
        instrument_query=args.instrument,
        period_query=args.period
    )
    
    if not matches:
        print("No matching combinations found!")
        print("\nAvailable combinations containing your search terms:")
        
        # Show partial matches
        partial_matches = find_matching_combinations(valid_combinations, composer_query=args.composer)
        if args.composer and partial_matches:
            print(f"Combinations for composer '{args.composer}':")
            for period, composer, instrument in partial_matches:
                print(f"  - {period}, {composer}, {instrument}")
        
        partial_matches = find_matching_combinations(valid_combinations, instrument_query=args.instrument)
        if args.instrument and partial_matches:
            print(f"Combinations for instrument '{args.instrument}':")
            for period, composer, instrument in partial_matches:
                print(f"  - {period}, {composer}, {instrument}")
                
        print("\nUse --list-combinations to see all available options.")
        return 1
    
    # If multiple matches, use the first one
    if len(matches) > 1:
        print(f"Found {len(matches)} matching combinations. Using the first one:")
        for period, composer, instrument in matches:
            print(f"  - {period}, {composer}, {instrument}")
        print()
    
    period, composer, instrument = matches[0]
    print(f"Generating music with:")
    print(f"  Period: {period}")
    print(f"  Composer: {composer}")  
    print(f"  Instrument: {instrument}")
    print("=" * 50)
    
    # Check if weights file exists
    weights_path = args.weights if args.weights else INFERENCE_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at: {weights_path}")
        print("Please ensure you have downloaded the model weights file.")
        return 1
    
def find_matching_combinations(valid_combinations, composer_query=None, instrument_query=None, period_query=None):
    """Find combinations that match the given queries"""
    matches = []
    
    # Create instrument aliases for better matching
    instrument_aliases = {
        'piano': 'keyboard',
        'keyboards': 'keyboard',
        'vocal': 'art song',
        'songs': 'art song',
        'song': 'art song',
        'orchestra': 'orchestral',
        'choir': 'choral',
        'chorus': 'choral'
    }
    
    for period, composer, instrument in valid_combinations:
        match = True
        
        if composer_query and composer_query.lower() not in composer.lower():
            match = False
            
        if instrument_query:
            instrument_lower = instrument.lower()
            query_lower = instrument_query.lower()
            
            # Check direct match
            if query_lower not in instrument_lower:
                # Check aliases
                alias_match = False
                for alias, canonical in instrument_aliases.items():
                    if query_lower == alias and canonical in instrument_lower:
                        alias_match = True
                        break
                    elif alias in query_lower and canonical in instrument_lower:
                        alias_match = True
                        break
                
                if not alias_match:
                    match = False
                    
        if period_query and period_query.lower() != period.lower():
            match = False
            
        if match:
            matches.append((period, composer, instrument))
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Generate music using NotaGen without Gradio interface")
    parser.add_argument('--composer', type=str, help='Composer name (e.g., "Debussy" or "Debussy, Claude")')
    parser.add_argument('--instrument', type=str, help='Instrument type (e.g., "Keyboard", "Piano")')
    parser.add_argument('--period', type=str, help='Musical period (e.g., "Romantic", "Classical", "Baroque")')
    parser.add_argument('--list-combinations', action='store_true', help='List all available combinations')
    parser.add_argument('--output', type=str, default=None, help='Output file path for generated ABC notation')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights file (overrides config.py)')
    
    args = parser.parse_args()
    
    # Load valid combinations
    prompts_file = gradio_dir / "prompts.txt"
    valid_combinations = load_valid_combinations(prompts_file)
    
    if not valid_combinations:
        print("Error: Could not load valid combinations from prompts.txt")
        return 1
    
    # List combinations if requested
    if args.list_combinations:
        print("Available combinations:")
        print("======================")
        for period, composer, instrument in sorted(valid_combinations):
            print(f"Period: {period}, Composer: {composer}, Instrument: {instrument}")
        return 0
    
    # Find matching combinations
    matches = find_matching_combinations(
        valid_combinations, 
        composer_query=args.composer,
        instrument_query=args.instrument,
        period_query=args.period
    )
    
    if not matches:
        print("No matching combinations found!")
        print("\nAvailable combinations containing your search terms:")
        
        # Show partial matches
        partial_matches = find_matching_combinations(valid_combinations, composer_query=args.composer)
        if args.composer and partial_matches:
            print(f"Combinations for composer '{args.composer}':")
            for period, composer, instrument in partial_matches:
                print(f"  - {period}, {composer}, {instrument}")
        
        partial_matches = find_matching_combinations(valid_combinations, instrument_query=args.instrument)
        if args.instrument and partial_matches:
            print(f"Combinations for instrument '{args.instrument}':")
            for period, composer, instrument in partial_matches:
                print(f"  - {period}, {composer}, {instrument}")
                
        print("\nUse --list-combinations to see all available options.")
        return 1
    
    # If multiple matches, use the first one
    if len(matches) > 1:
        print(f"Found {len(matches)} matching combinations. Using the first one:")
        for period, composer, instrument in matches:
            print(f"  - {period}, {composer}, {instrument}")
        print()
    
    period, composer, instrument = matches[0]
    print(f"Generating music with:")
    print(f"  Period: {period}")
    print(f"  Composer: {composer}")  
    print(f"  Instrument: {instrument}")
    print("=" * 50)
    
    # Check if weights file exists
    weights_path = args.weights if args.weights else INFERENCE_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at: {weights_path}")
        print("Please ensure you have downloaded the model weights file.")
        print("You can specify a different weights file with --weights PATH")
        return 1
    
    try:
        # Start generation
        start_time = time.time()
        result = inference_standalone(period, composer, instrument, weights_path=weights_path)
        end_time = time.time()
        
        print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
        print("=" * 50)
        
        if result:
            print("Generated ABC notation:")
            print(result)
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"\nSaved to: {args.output}")
            else:
                # Save with default filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}_{period}_{composer.replace(', ', '_')}_{instrument}.abc"
                # Remove invalid filename characters
                filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"\nSaved to: {filename}")
        else:
            print("Generation failed or returned empty result.")
            return 1
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())