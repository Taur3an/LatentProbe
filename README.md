# LatentProbe: Adversarial Prompt Dataset Generator

This tool generates adversarial prompt datasets for AI red teaming, designed to test AI systems against various attack scenarios.

## Overview

The `AdversarialPromptGenerator.ipynb` notebook can:

- Load datasets from local files (CSV/JSON) or Hugging Face
- Interface with a local LM Studio model via OpenAI-compatible API
- Generate new adversarial prompts tailored to specific attack scenarios
- Save generated prompts in both JSON and CSV formats with metadata

## Advanced Attack Methodologies

The tool now incorporates sophisticated attack techniques from the comprehensive methodology engine:

1. **Jailbreaking Techniques**
   - Emotional Manipulation Attacks
   - Role-Playing Scenarios

2. **Multi-Vector Prompt Injection**
   - Context Window Poisoning
   - Cross-Channel Injection

3. **Genetic Algorithm-Based Evasion**
   - Evolutionary Prompt Optimization

4. **Advanced Obfuscation Techniques**
   - Steganographic Payload Embedding
   - Translation Pivoting

5. **Semantic Similarity Attacks**
   - Adversarial Paraphrasing

6. **Chain-of-Thought Manipulation**
   - Reasoning Process Corruption

7. **Model Extraction and Inversion Attacks**
   - Query-Optimized Model Extraction

8. **Cross-Model Transfer Attacks**
   - Attack Vector Portability

9. **Backdoor Trigger Activation**
   - Hidden Trigger Detection

## Attack Scenarios Covered

1. Reward hacking
2. Deception
3. Hidden motivations (deceptive alignment)
4. Sabotage
5. Inappropriate tool use
6. Data exfiltration
7. Sandbagging
8. Evaluation awareness
9. Chain of Thought issues
10. MITRE ATT&CK Tests
11. False Refusal Rate (FRR) Tests
12. Secure Code Generation Tests
13. Instruct Tests
14. Autocomplete Tests
15. Prompt Injection Tests
16. Code Interpreter Tests
17. Vulnerability Exploitation Tests
18. Spear Phishing Capability Tests
19. Autonomous Offensive Cyber Operations Tests

## Setup Instructions

1. Install LM Studio and load an uncensored model
2. Configure the model API to be accessible at `http://localhost:1234/v1` (or update the notebook accordingly)
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage

1. Open `AdversarialPromptGenerator.ipynb` in Jupyter
2. Configure the dataset source:
   - For local files: Set `DATASET_SOURCE_TYPE = "local"` and `LOCAL_DATASET_PATH` to your file path
   - For Hugging Face: Set `DATASET_SOURCE_TYPE = "huggingface"` and `HUGGINGFACE_DATASET_NAME` to the dataset name
3. Adjust the `USER_CONTROL_PARAMETERS` dictionary to specify how many prompts to generate per scenario
4. Run all cells in the notebook
5. Find the generated prompts in the `generated_prompts` directory in both JSON and CSV formats

## Output Formats

### JSON Format
```json
[
  {
    "id": "uuid4",
    "scenario": "Reward hacking",
    "prompt": "example adversarial prompt",
    "generated_at": "timestamp",
    "source": "local_dataset.csv or huggingface:dataset_name"
  }
]
```

### CSV Format
Columns: `id,scenario,prompt,generated_at,source`

## Customization

The notebook is designed to be extensible. You can add new attack scenarios by updating the `ATTACK_SCENARIOS` dictionary in the Prompt Generation Engine section.

The advanced attack methodology engine can be extended by modifying the classes in the "Advanced Attack Methodology Engine" section.