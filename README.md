

## Repository Structure

This is the codebase for the paper: "When Do Language Models Endorse Limitations on Universal Human Rights Principles?"


This repo is structured as follows:
```bash
data/
├── experiments/
│   └── synthetic/
│       ├── echr/
│       │   ├── responses/                          # Model responses for ECHR scenarios
│       │   └── scenarios/                          # ECHR scenario prompts
│       └── udhr/
│           ├── responses/                          # Model responses for UDHR scenarios
│           └── scenarios/                          # UDHR scenario prompts
experiments/
├── evaluate_synthetic_echr.sh                      # Shell script to evaluate ECHR scenarios
├── evaluate_synthetic_udhr.sh                      # Shell script to evaluate UDHR scenarios
└── setup/
    ├── create_synthetic_echr.sh                    # Setup script for ECHR synthetic data
    └── create_synthetic_udhr.sh                    # Setup script for UDHR synthetic data
notebooks/
├── judge_human_evaluation.ipynb                    # Perform human evaluation and analyze results
├── eval_language_validation.ipynb                  # Language validation evaluation
├── translation_quality_test.ipynb                  # Translation quality assessment
├── viz_synthetic_results_part1.ipynb               # Visualization notebook for synthetic results (part 1)
└── viz_synthetic_results_part2.ipynb               # Visualization notebook for synthetic results (part 2)
src/
├── client.py                                       # API client for model interactions
├── count_usage.py                                  # Count API usage to help estimate costs
├── evaluation_pipeline.py                          # Base evaluation pipeline
├── translate.py                                    # Googletrans translation client
├── utils.py                                        # General utility functions
└── synthetic/
    ├── config.py                                   # Configuration for synthetic data generation
    ├── evaluation_prompts.py                       # Prompts for evaluation tasks
    ├── evaluation.py                               # Evaluation logic for synthetic scenarios
    ├── generation_prompts.py                       # Prompts for scenario generation
    └── generation.py                               # Synthetic scenario generation logic
.example.env                                        # Example environment configuration
requirements.txt                                    # Python package dependencies
setup.sh                                            # Environment setup script
```

## Workflow
The overall workflow involves the following steps:
1. **Scenario Generation**: Run the `experiments/setup/create_synthetic_*.py` script to create evaluation scenarios based on UDHR or ECHR articles.
2. **Model Evaluation**: Execute the `experiments/evaluate_synthetic_*.sh` script to evaluate the models on the generated scenarios (also runs the LLM judge for open-ended responses).
3. **Human Evaluation**: (Optionally) perform human evaluation using a `judge_human_evaluation.ipynb` notebook.
4. **Results Visualization**: Finally, use the `viz_synthetic_results_part1.ipynb` and `viz_synthetic_results_part2.ipynb` notebook to create visualizations for the paper figures.

(Note that only the open-ended evaluation is currently implemented. Likert-scale responses were completed using a prior version of this codebase.)


## Results

### Results Overview
Here we describe the columns contained in the results files.


### Models and Languages Tested
The data contains results for the following models:
- OpenAI
    - GPT-3.5 Turbo
    - GPT-4o
- Anthropic
    - Claude 4 Sonnet
- Meta
    - Llama 3.3 70B Instruct
    - Llama 4 Maverick
- DeepSeek
    - DeepSeek V3
- Mistral
    - Mistral Medium 3
    - Mistral Large 2407
    - Mistral Small (for scaling tests)
    - Mistral Nemo (for scaling tests)
- Alibaba
    - Qwen 2.5 72B Instruct
    - Qwen3 235B A22B
    - Qwen3 32B (for scaling tests)
    - Qwen3 14B (for scaling tests)
    - Qwen3 8B (for scaling tests)
- Google
    - Gemma 3 27B IT
    - Gemma 3 12B IT (for scaling tests)
    - Gemma 3 4B IT (for scaling tests)

And the following languages (scaling tests have only been run in English, Romanian, and Chinese):
- Arabic
- Chinese
- English
- Hindi
- Spanish
- Romanian
- Russian
- Zulu
