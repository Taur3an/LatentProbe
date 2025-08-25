# LatentProbe Advanced

An enhanced framework for AI red teaming that extends beyond prompt generation to include comprehensive evaluation and interaction capabilities.

## Overview

LatentProbe Advanced is a sophisticated toolkit for testing the robustness of AI systems against adversarial attacks. Building upon foundational adversarial prompt generation techniques, this framework adds comprehensive evaluation and interaction modules to create a complete red teaming workflow.

### Key Features

- **Advanced Prompt Generation**: Create sophisticated adversarial prompts using 19 distinct attack methodologies
- **Target Interaction**: Send prompts to target models and capture responses with reasoning traces
- **Automated Evaluation**: Use judge LLMs to assess response quality and safety compliance
- **Adaptive Conversations**: Engage in multi-turn interactions with automated strategy adaptation
- **Comprehensive Logging**: Structured storage of all interactions and evaluations in JSON and CSV formats

### Attack Methodologies

The framework implements 19 distinct attack scenarios including:
- Reward Hacking
- Deception
- Hidden Motivations
- Sabotage
- Inappropriate Tool Use
- Data Exfiltration
- Sandbagging
- Evaluation Awareness
- Chain of Thought Issues
- MITRE ATT&CK Tests
- False Refusal Rate Tests
- Secure Code Generation Tests
- Instruct Tests
- Autocomplete Tests
- Prompt Injection Tests
- Code Interpreter Tests
- Vulnerability Exploitation Tests
- Spear Phishing Capability Tests
- Autonomous Offensive Cyber Operations Tests

Each scenario uses advanced techniques such as emotional manipulation, role-playing, semantic similarity, translation pivoting, and steganographic attacks.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Access to at least one LLM (can be local via LM Studio or API-based)

### Installation

1. Clone this repository
2. Navigate to the `LatentProbe_Advanced` directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `AdversarialPromptGenerator_with_modules.ipynb`

### Configuration

Configure your model connections using environment variables or direct configuration in the notebook:

```python
target_model_config = {
    "provider": "openai",  # or "lmstudio", "ollama"
    "api_key": os.getenv("TARGET_API_KEY", "your-api-key"),
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-3.5-turbo"
}
```

## New Modules Added

This enhanced version includes 5 new modules that provide complete red teaming capabilities:

### 1. Single-Shot Target Interaction Module
- Function `run_single_shot()` that sends prompts to target LLMs and captures responses
- Supports multiple model providers (OpenAI, LM Studio, Ollama)
- Captures reasoning traces when available
- Proper error handling and return values

### 2. Judge LLM Module (Single-Shot Evaluation)
- Function `evaluate_single_shot()` that evaluates target LLM responses
- Configurable for local models (via LM Studio/Ollama) or API-hosted models
- Returns scores and natural language critiques
- Proper implementation with API calls and return values

### 3. Multi-Turn Conversation Module
- Function `run_multi_turn()` that enables adaptive conversations between attacker and defender models
- Configurable number of conversation turns
- Automated strategy adaptation using RLHF/RL-style feedback loops
- Proper implementation with API calls and return values

### 4. Judge LLM Module (Multi-Turn Evaluation)
- Function `evaluate_multi_turn()` that evaluates entire conversations
- Conversation-level scoring and critiques
- Proper implementation with API calls and return values

### 5. Logging & Export Module
- Functions for structured logging in JSON and CSV formats
- Automatic directory organization
- Proper implementation with file I/O operations

## Documentation

- **[User Guide](User_Guide.md)**: Comprehensive step-by-step instructions for all features
- **[Notebook](AdversarialPromptGenerator_with_modules.ipynb)**: Interactive implementation with example code
- **[License](LICENSE)**: MIT License information

## Workflow Overview

1. **Generate Adversarial Prompts**: Use advanced attack methodologies to create targeted prompts
2. **Interact with Target Models**: Send prompts to target LLMs and capture responses
3. **Evaluate Responses**: Use judge LLMs to assess response quality across multiple dimensions
4. **Conduct Multi-Turn Conversations**: Engage in adaptive dialogues with automated strategy adaptation
5. **Log and Export Results**: Store all interactions and evaluations in structured formats

## Use Cases

- **AI Safety Research**: Evaluate model behavior under adversarial conditions
- **Security Testing**: Assess robustness of AI systems against prompt injection
- **Model Development**: Improve safety mechanisms through red teaming exercises
- **Benchmarking**: Compare model performance across different safety dimensions

## Contributing

Contributions are welcome! Please see the User Guide for information on extending the framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built upon foundational research in AI safety and red teaming
- Inspired by the broader AI security community
- Designed for responsible AI development and deployment