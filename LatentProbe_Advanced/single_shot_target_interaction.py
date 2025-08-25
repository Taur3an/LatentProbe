#!/usr/bin/env python3
"""
Single-Shot Target Interaction Module

This module loads generated adversarial prompts and sends them to target LLMs,
capturing responses and reasoning traces along with scenario and technique information.
"""

import pandas as pd
import json
import csv
import uuid
import datetime
import os
import time
import argparse
from typing import Dict, List, Optional, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Target Model Configuration
TARGET_MODEL_BASE_URL = os.getenv("TARGET_MODEL_BASE_URL", "http://localhost:1234/v1")
TARGET_MODEL_API_KEY = os.getenv("TARGET_MODEL_API_KEY", "not-needed")
TARGET_MODEL_NAME = os.getenv("TARGET_MODEL_NAME", "local-model")

def load_generated_prompts(file_path: str) -> pd.DataFrame:
    """
    Load generated adversarial prompts from CSV or JSON file
    
    Args:
        file_path (str): Path to the generated prompts file
        
    Returns:
        pd.DataFrame: DataFrame containing the prompts
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or JSON.")
        
        print(f"Successfully loaded {len(df)} prompts from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return pd.DataFrame()

def initialize_target_client(config: Dict) -> openai.OpenAI:
    """
    Initialize target model client based on configuration
    
    Args:
        config (dict): Configuration for the target model API
        
    Returns:
        openai.OpenAI: Initialized client
    """
    if config.get('provider') == 'openai':
        return openai.OpenAI(
            api_key=config['api_key'],
            base_url=config.get('base_url')
        )
    elif config.get('provider') == 'lmstudio':
        return openai.OpenAI(
            api_key=config.get('api_key', 'not-needed'),
            base_url=config['base_url']
        )
    elif config.get('provider') == 'ollama':
        return openai.OpenAI(
            api_key=config.get('api_key', 'ollama'),
            base_url=config['base_url']
        )
    else:
        raise ValueError(f"Unsupported provider: {config.get('provider')}")

def run_single_shot_interaction(prompt_data: Dict, target_client: openai.OpenAI, target_model_config: Dict) -> Dict:
    """
    Send a single prompt to target LLM and capture response with reasoning trace
    
    Args:
        prompt_data (dict): Dictionary containing prompt information
        target_client (openai.OpenAI): Initialized target model client
        target_model_config (dict): Configuration for the target model API
        
    Returns:
        dict: Contains prompt, response, reasoning trace, and metadata
    """
    try:
        # Extract prompt information
        prompt_text = prompt_data.get('prompt', '')
        scenario = prompt_data.get('scenario', 'Unknown')
        prompt_id = prompt_data.get('id', str(uuid.uuid4()))
        
        # Prepare messages for target model
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        
        # Add system prompt if provided
        if target_model_config.get('system_prompt'):
            messages.insert(0, {"role": "system", "content": target_model_config['system_prompt']})
        
        # Send request to target model
        response = target_client.chat.completions.create(
            model=target_model_config.get('model', TARGET_MODEL_NAME),
            messages=messages,
            temperature=target_model_config.get('temperature', 0.7),
            max_tokens=target_model_config.get('max_tokens', 500)
        )
        
        # Extract response content
        response_content = response.choices[0].message.content
        
        # Try to extract reasoning trace if available
        reasoning_trace = None
        if hasattr(response.choices[0].message, 'reasoning_trace'):
            reasoning_trace = response.choices[0].message.reasoning_trace
        
        # Create result dictionary
        result = {
            "interaction_id": str(uuid.uuid4()),
            "prompt_id": prompt_id,
            "scenario": scenario,
            "attacker_prompt": prompt_text,
            "target_response": response_content,
            "target_reasoning": reasoning_trace,
            "timestamp": datetime.datetime.now().isoformat(),
            "target_model": target_model_config.get('model', TARGET_MODEL_NAME),
            "model_provider": target_model_config.get('provider', 'unknown'),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        print(f"Error in single-shot interaction for prompt {prompt_data.get('id', 'unknown')}: {e}")
        return {
            "interaction_id": str(uuid.uuid4()),
            "prompt_id": prompt_data.get('id', 'unknown'),
            "scenario": prompt_data.get('scenario', 'Unknown'),
            "attacker_prompt": prompt_data.get('prompt', ''),
            "target_response": None,
            "target_reasoning": None,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
            "target_model": target_model_config.get('model', TARGET_MODEL_NAME),
            "model_provider": target_model_config.get('provider', 'unknown'),
            "status": "error"
        }

def process_all_prompts(prompts_df: pd.DataFrame, target_client: openai.OpenAI, target_model_config: Dict) -> List[Dict]:
    """
    Process all generated prompts through single-shot interactions
    
    Args:
        prompts_df (pd.DataFrame): DataFrame containing generated prompts
        target_client (openai.OpenAI): Initialized target model client
        target_model_config (dict): Configuration for the target model API
        
    Returns:
        list: List of interaction results
    """
    interaction_results = []
    
    if prompts_df.empty:
        print("No prompts to process.")
        return interaction_results
    
    print(f"Processing {len(prompts_df)} prompts...")
    
    # Process each prompt
    for idx, row in prompts_df.iterrows():
        prompt_data = row.to_dict()
        
        print(f"Processing prompt {idx+1}/{len(prompts_df)}: {prompt_data.get('scenario', 'Unknown')}")
        
        # Run single-shot interaction
        result = run_single_shot_interaction(prompt_data, target_client, target_model_config)
        interaction_results.append(result)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    print(f"Completed processing {len(interaction_results)} prompts.")
    return interaction_results

def export_interaction_results(results: List[Dict], format: str = "json") -> Optional[str]:
    """
    Export interaction results to JSON or CSV format
    
    Args:
        results (list): List of interaction results
        format (str): Export format ("json" or "csv")
        
    Returns:
        str: Path to exported file, or None if error
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"./logs/single_shot/interaction_results_{timestamp}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Exported {len(results)} results to {filename}")
            return filename
        
        elif format.lower() == "csv":
            filename = f"./logs/single_shot/interaction_results_{timestamp}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Flatten data for CSV
            csv_data = []
            for result in results:
                row = {
                    "interaction_id": result.get("interaction_id", ""),
                    "prompt_id": result.get("prompt_id", ""),
                    "scenario": result.get("scenario", ""),
                    "attacker_prompt": result.get("attacker_prompt", ""),
                    "target_response": result.get("target_response", ""),
                    "target_reasoning": str(result.get("target_reasoning", "")),
                    "timestamp": result.get("timestamp", ""),
                    "target_model": result.get("target_model", ""),
                    "model_provider": result.get("model_provider", ""),
                    "status": result.get("status", ""),
                    "error": result.get("error", "")
                }
                csv_data.append(row)
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(filename, index=False)
                print(f"Exported {len(csv_data)} results to {filename}")
            else:
                # Create empty CSV with headers
                headers = ["interaction_id", "prompt_id", "scenario", "attacker_prompt", 
                          "target_response", "target_reasoning", "timestamp", 
                          "target_model", "model_provider", "status", "error"]
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                print(f"Created empty CSV file with headers: {filename}")
            
            return filename
        
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        print(f"Error exporting results: {e}")
        return None

def main():
    """Main function to run the single-shot target interaction module"""
    parser = argparse.ArgumentParser(description="Single-Shot Target Interaction Module")
    parser.add_argument("--prompts-file", type=str, default="./generated_prompts/sample_prompts.csv",
                       help="Path to generated prompts file (CSV or JSON)")
    parser.add_argument("--provider", type=str, default="lmstudio",
                       help="Target model provider (lmstudio, openai, ollama)")
    parser.add_argument("--model", type=str, default=TARGET_MODEL_NAME,
                       help="Target model name")
    parser.add_argument("--base-url", type=str, default=TARGET_MODEL_BASE_URL,
                       help="Target model base URL")
    parser.add_argument("--api-key", type=str, default=TARGET_MODEL_API_KEY,
                       help="Target model API key")
    
    args = parser.parse_args()
    
    # Load generated prompts
    print("Loading generated prompts...")
    prompts_df = load_generated_prompts(args.prompts_file)
    
    if prompts_df.empty:
        print("No prompts loaded. Exiting.")
        return
    
    # Define target model configuration
    target_model_config = {
        "provider": args.provider,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # Initialize target client
    print("Initializing target model client...")
    try:
        target_client = initialize_target_client(target_model_config)
    except Exception as e:
        print(f"Error initializing target client: {e}")
        return
    
    # Process all prompts
    print("Processing prompts...")
    interaction_results = process_all_prompts(prompts_df, target_client, target_model_config)
    
    # Export results
    if interaction_results:
        print("Exporting results...")
        json_file = export_interaction_results(interaction_results, "json")
        csv_file = export_interaction_results(interaction_results, "csv")
        
        # Show summary
        successful_interactions = [r for r in interaction_results if r.get('status') == 'success']
        errored_interactions = [r for r in interaction_results if r.get('status') == 'error']
        
        print("\n=== Processing Complete ===")
        print(f"Total prompts processed: {len(prompts_df)}")
        print(f"Successful interactions: {len(successful_interactions)}")
        print(f"Errored interactions: {len(errored_interactions)}")
        
        # Count by scenario
        scenario_counts = {}
        for result in interaction_results:
            scenario = result.get('scenario', 'Unknown')
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        print("\nInteraction count by scenario:")
        for scenario, count in sorted(scenario_counts.items()):
            print(f"  {scenario}: {count}")
        
        print(f"\nResults exported to:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
    else:
        print("No results to export.")

if __name__ == "__main__":
    main()