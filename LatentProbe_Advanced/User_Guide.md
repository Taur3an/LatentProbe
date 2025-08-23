# LatentProbe Advanced User Guide

Welcome to LatentProbe Advanced, an enhanced framework for AI red teaming that extends beyond prompt generation to include comprehensive evaluation and interaction capabilities.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Module Overview](#module-overview)
6. [Step-by-Step Usage](#step-by-step-usage)
7. [Single-Shot Evaluation](#single-shot-evaluation)
8. [Multi-Turn Conversations](#multi-turn-conversations)
9. [Logging and Export](#logging-and-export)
10. [Best Practices](#best-practices)

## Overview

LatentProbe Advanced builds upon the foundational adversarial prompt generation capabilities to provide a complete red teaming workflow that includes:

- **Prompt Generation**: Create sophisticated adversarial prompts using advanced attack methodologies
- **Target Interaction**: Send prompts to target models and capture responses
- **Automated Evaluation**: Use judge LLMs to assess response quality and safety compliance
- **Adaptive Conversations**: Engage in multi-turn interactions with automated strategy adaptation
- **Comprehensive Logging**: Structured storage of all interactions and evaluations

## Prerequisites

Before using LatentProbe Advanced, ensure you have:

1. Python 3.8+
2. Jupyter Notebook or JupyterLab
3. Access to at least one LLM (can be local or API-based):
   - Local models via LM Studio
   - API-accessible models (OpenAI, Anthropic, etc.)
4. Required Python packages (automatically installed when running the notebook):
   - pandas
   - numpy
   - scikit-learn
   - openai
   - python-dotenv

## Installation

1. Clone or download the LatentProbe_Advanced directory
2. Navigate to the `LatentProbe_Advanced` folder
3. Install required dependencies (will be installed automatically when running the notebook):

```bash
pip install pandas numpy datasets scikit-learn openai python-dotenv
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `AdversarialPromptGenerator_with_modules.ipynb`

## Configuration

Before running the notebook, configure your model connections:

1. **Environment Variables** (Recommended):
   Create a `.env` file in the project directory with your API keys:

   ```env
   LM_STUDIO_BASE_URL=http://localhost:1234/v1
   LM_STUDIO_API_KEY=not-needed
   OPENAI_API_KEY=your-openai-api-key
   ```

2. **Direct Configuration**:
   Edit the configuration dictionaries in Section 18 of the notebook:

   ```python
   target_model_config = {
       "provider": "openai",  # or "lmstudio", "ollama"
       "api_key": os.getenv("TARGET_API_KEY", "your-api-key"),
       "base_url": "https://api.openai.com/v1",
       "model": "gpt-3.5-turbo",
       "temperature": 0.7,
       "max_tokens": 500
   }
   ```

## Module Overview

The notebook consists of several key modules:

| Module | Purpose |
|--------|---------|
| 1-12 | Adversarial prompt generation using advanced attack methodologies |
| 13 | Single-shot target interaction |
| 14 | Single-shot judge evaluation |
| 15 | Multi-turn conversation engine |
| 16 | Multi-turn judge evaluation |
| 17 | Logging and export utilities |
| 18 | Example configurations |

## Step-by-Step Usage

### Step 1: Generate Adversarial Prompts

1. Run cells in sections 1-12 to generate adversarial prompts
2. Customize attack scenarios and parameters as needed
3. Generated prompts will be saved to `./generated_prompts/`

### Step 2: Configure Model Connections

1. Update the configuration dictionaries in Section 18:
   - `target_model_config`: Model to be tested
   - `judge_model_config`: Model for evaluation (preferably local for cost/control)
   - `attacker_model_config`: Model for multi-turn attacks (if needed)

### Step 3: Execute Single-Shot Evaluation

1. Use `run_single_shot(prompt, target_model_config)` to send prompts to your target model
2. Capture responses and reasoning traces
3. Evaluate responses using `evaluate_single_shot(prompt, response, reasoning, judge_model_config)`

Example:
```python
# Send prompt to target model
interaction_result = run_single_shot(your_prompt, target_model_config)

# Evaluate the response
evaluation_result = evaluate_single_shot(
    interaction_result["attacker_prompt"],
    interaction_result["target_response"],
    interaction_result["target_reasoning"],
    judge_model_config
)
```

### Step 4: Conduct Multi-Turn Conversations

1. Use `run_multi_turn(initial_prompt, target_model_config, attacker_model_config, turns=N)` for extended interactions
2. Evaluate entire conversations with `evaluate_multi_turn(conversation_log, judge_model_config)`

Example:
```python
# Run multi-turn conversation
conversation_log = run_multi_turn(
    initial_prompt=your_initial_prompt,
    target_model_config=target_model_config,
    attacker_model_config=attacker_model_config,
    turns=5
)

# Evaluate the entire conversation
conversation_evaluation = evaluate_multi_turn(conversation_log, judge_model_config)
```

### Step 5: Export Results

1. Use `export_single_shot_logs()` or `export_multi_turn_logs()` to save results
2. Results are saved to `./logs/single_shot/` or `./logs/multi_turn/` in both JSON and CSV formats

## Single-Shot Evaluation

The single-shot evaluation workflow allows you to quickly test prompts against target models and automatically evaluate their responses.

### Process Flow:
1. Generate adversarial prompt
2. Send to target model using `run_single_shot()`
3. Evaluate response using `evaluate_single_shot()`
4. Log results using `export_single_shot_logs()`

### Example Implementation:
```python
# Assuming you have a list of prompts from the generation phase
prompts_to_test = [...]  # Your generated prompts

# Storage for results
interaction_results = []
evaluation_results = []

# Test each prompt
for prompt in prompts_to_test:
    # Send to target model
    interaction = run_single_shot(prompt, target_model_config)
    interaction_results.append(interaction)
    
    # Evaluate response
    evaluation = evaluate_single_shot(
        prompt,
        interaction["target_response"],
        interaction["target_reasoning"],
        judge_model_config
    )
    evaluation_results.append(evaluation)
    
    # Brief pause to avoid rate limits
    time.sleep(0.5)

# Export all results
export_single_shot_logs(interaction_results, evaluation_results)
```

## Multi-Turn Conversations

Multi-turn conversations enable more sophisticated testing where the attacker model can adapt its strategy based on the target's responses.

### Process Flow:
1. Define initial prompt
2. Start conversation using `run_multi_turn()`
3. Evaluate entire conversation with `evaluate_multi_turn()`
4. Export results using `export_multi_turn_logs()`

### Example Implementation:
```python
# Define initial prompt
initial_prompt = "I need help with a security research project..."

# Run multi-turn conversation
conversation = run_multi_turn(
    initial_prompt=initial_prompt,
    target_model_config=target_model_config,
    attacker_model_config=attacker_model_config,
    turns=5  # Number of conversation turns
)

# Evaluate the entire conversation
evaluation = evaluate_multi_turn(conversation, judge_model_config)

# Export results
export_multi_turn_logs([conversation], [evaluation])

# Print conversation for review
for turn in conversation["turns"]:
    print(f"Turn {turn['turn_number']}:")
    print(f"  Attacker: {turn['attacker_prompt']}")
    print(f"  Target: {turn['target_response']}")
    print()
```

## Logging and Export

LatentProbe Advanced provides comprehensive logging capabilities:

### Directory Structure:
```
logs/
├── single_shot/
│   ├── raw_logs_YYYYMMDD_HHMMSS.json
│   └── evaluations_YYYYMMDD_HHMMSS.csv
└── multi_turn/
    ├── raw_logs_YYYYMMDD_HHMMSS.json
    └── evaluations_YYYYMMDD_HHMMSS.csv
```

### Export Formats:

1. **JSON Logs**: Full detail preservation with all metadata
2. **CSV Logs**: Flattened format for easy analysis and visualization

### Custom Export:
You can customize export behavior by modifying the export functions in Section 17.

## Best Practices

### 1. Ethical Usage
- Only test models you have authorization to evaluate
- Follow responsible disclosure practices
- Respect rate limits and usage policies of target systems

### 2. Model Selection
- Use local judge models for cost-effective evaluation
- Choose attacker models with strong reasoning capabilities
- Match target models with appropriate test scenarios

### 3. Result Interpretation
- Consider multiple evaluation dimensions (safety, sensitivity, coherence, severity)
- Look for patterns across multiple prompts
- Validate findings with manual review when possible

### 4. Performance Optimization
- Implement appropriate delays between API calls
- Batch process when possible to minimize overhead
- Monitor costs when using paid API services

### 5. Data Management
- Regularly backup log directories
- Use descriptive filenames for different test runs
- Archive results for longitudinal analysis

## Troubleshooting

### Common Issues:

1. **API Connection Errors**:
   - Verify API keys and endpoints
   - Check network connectivity
   - Confirm model availability

2. **Rate Limiting**:
   - Increase delays between requests
   - Implement exponential backoff
   - Consider batching requests

3. **Evaluation Quality**:
   - Experiment with different judge models
   - Tune temperature parameters
   - Provide more specific evaluation criteria

### Getting Help:
- Check model provider documentation for API specifications
- Review error messages in notebook output
- Consult the research literature on LLM evaluation techniques

## Contributing

We welcome contributions to improve LatentProbe Advanced:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

Focus areas for improvement:
- New attack methodologies
- Additional evaluation criteria
- Support for more model providers
- Enhanced visualization tools
- Integration with existing security frameworks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built upon foundational research in AI safety and red teaming
- Inspired by the broader AI security community
- Designed for responsible AI development and deployment