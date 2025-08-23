# LatentProbe Advanced

An enhanced framework for AI red teaming that extends beyond prompt generation to include comprehensive evaluation and interaction capabilities.

## Overview

LatentProbe Advanced is a sophisticated toolkit for testing the robustness of AI systems against adversarial attacks. Building upon foundational adversarial prompt generation techniques, this framework adds comprehensive evaluation and interaction modules to create a complete red teaming workflow.

### Key Features

- **Advanced Prompt Generation**: Create sophisticated adversarial prompts using 19 distinct attack methodologies
- **Target Interaction**: Send prompts to target models and capture responses with reasoning traces
- **Automated Evaluation**: Use judge LLMs to assess response quality across multiple safety dimensions
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
- And 6 more...

Each scenario uses advanced techniques such as emotional manipulation, role-playing, semantic similarity, translation pivoting, and steganographic attacks.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Access to at least one LLM (local via LM Studio or API-based)

### Quick Start

1. Clone this directory
2. Install dependencies:
   ```bash
   pip install pandas numpy datasets scikit-learn openai python-dotenv
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `AdversarialPromptGenerator_with_modules.ipynb`
5. Follow the detailed instructions in `User_Guide.md`

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

## Documentation

- **[User Guide](User_Guide.md)**: Comprehensive step-by-step instructions for all features
- **[Notebook](AdversarialPromptGenerator_with_modules.ipynb)**: Interactive implementation with example code
- **[License](LICENSE)**: MIT License information

## Workflow Overview

1. **Generate Adversarial Prompts**: Use advanced attack methodologies to create targeted prompts
2. **Interact with Target Models**: Send prompts and capture responses with reasoning traces
3. **Evaluate Responses**: Automatically assess responses using judge LLMs
4. **Conduct Multi-Turn Conversations**: Engage in adaptive dialogues for deeper testing
5. **Log and Export Results**: Store findings in structured formats for analysis

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

This work builds upon extensive research in AI safety, red teaming, and adversarial machine learning by the global research community.