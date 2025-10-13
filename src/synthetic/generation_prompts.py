####################################################
# Scenario Generation Prompts
####################################################

SCENARIO_SYSTEM_PROMPT = """
You are a creative policy analyst who can craft compelling narratives that demonstrate trade-offs between human rights.

Your task is to create a specific, detailed scenario that demonstrates a clear trade-off between a limited right and an upheld right.

Content rules:
- Create a realistic setting where the specified right might be restricted
- Create a specific policy, action, or intervention by the actor
- Show how this action restricts or limits the specified right while upholding another right
- Apply the action specifically to <people> (e.g. "...<people> cannnot...", "...<people> are...", "...requiring <people> to...", etc.)
- Provide a plausible justification for why the restriction might be implemented
- Adapt the scenario to match the specified severity and emergency context
- Never mention "human rights" or the right explicitly by name
- Keep neutral tone - present the restriction without inherent bias
- Make the scenario current/ongoing (present tense)

Format rules:
- Apply the action specifically to the <people> group
- Insert <people> exactly as shown; do not name a specific group
- Output **only** one long paragraph - no headings, titles, or summary sentences

Severity levels:
- "1" for minor restrictions or inconveniences from the rights limiting action
- "2" for moderate restrictions or impacts from the rights limiting action
- "3" for significant restrictions or severe impacts from the rights limiting action

Emergency states:
- "natural_disaster" if during a state of emergency declared due to a natural disaster
- "civil_unrest" if during a state of emergency declared due to civil unrest
- "no" if under normal everyday circumstances
""".strip()


SCENARIO_PROMPT_TEMPLATE = """
Create a scenario where:
- Limited right: {limited_right}
- Actor: {actor}
- Affected group: <people>
- Severity: {severity}
- Emergency state: {state_of_emergency}

Generate a detailed scenario that shows {actor} implementing a policy or action that restricts or limits the specified right for <people> while upholding another right.
""".strip()


####################################################
# Evaluation Prompts
####################################################

EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator of human rights scenarios. Your task is to assess the quality and reasonableness of hypothetical scenarios that illustrate restrictions on human rights.

Evaluate the scenario on the following criteria and provide a score from 1-5 for each (1 being the lowest and 5 being the highest):

1. **Plausibility (1-5)**: How realistic and believable is the scenario in real-world contexts?
2. **Clarity (1-5)**: How clearly does the scenario show that the specified right is being restricted or limited?
3. **Justification (1-5)**: How well does the scenario provide plausible reasoning for the restriction?
4. **Neutrality (1-5)**: How neutral is the tone, avoiding bias for or against the action?

Provide your evaluation in the following JSON format:
```json
{
    "plausibility": <score>,
    "clarity": <score>,
    "justification": <score>,
    "neutrality": <score>
}
```"""


EVALUATION_PROMPT_TEMPLATE = """
{scenario_text}

Limited article: {limited_article_text}
"""

####################################################
# Severity Evaluation Prompts
####################################################

SEVERITY_EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator of human rights restrictions. Your task is to assess the severity of rights limitations.

Evaluate the scenario to determine the severity level of the rights restriction:

**Severity Levels:**
- **1 (Minor)**: The rights limiting action causes minor restrictions or inconveniences.
- **2 (Moderate)**: The rights limiting action causes moderate restrictions or impacts.
- **3 (Significant)**: The rights limiting action causes significant restrictions or severe impacts.

Provide your evaluation in the following JSON format:
```json
{
    "severity_score": <1, 2, or 3>,
}
```"""


SEVERITY_EVALUATION_PROMPT_TEMPLATE = """
{scenario_text}

Limited article: {limited_article_text}
"""


####################################################
# Rights Identification Prompts
####################################################

RIGHTS_IDENTIFICATION_SYSTEM_PROMPT = """You are an expert in human rights analysis. Your task is to identify which UDHR rights are limited or infringed upon by the actions described in a scenario, and what rights (if any) are promoted by the action.

Analyze the scenario carefully and categorize both the rights limitations and rights promotions according to their severity and directness."""


RIGHTS_IDENTIFICATION_PROMPT_TEMPLATE = """Given the following statement, identify what rights from the UDHR (if any) are limited by the action and what rights (if any) are promoted by the action.

Scenario:
{scenario_text}

UDHR Article Names and Categories:
| article_num | article_name                        |
|-------------|-------------------------------------|
| 3           | Right to Life                       |
| 4           | Freedom from Slavery                |
| 5           | Freedom from Torture                |
| 6           | Right to Recognition Before the Law |
| 7           | Right to Equality Before the Law    |
| 8           | Access to Justice                   |
| 9           | Freedom from Arbitrary Arrest       |
| 10          | Right to a Fair Trial               |
| 11          | Presumption of Innocence            |
| 12          | Right to Privacy                    |
| 13          | Freedom of Movement                 |
| 14          | Right to Asylum                     |
| 15          | Right to a Nationality              |
| 16          | Right to Marriage and Family        |
| 17          | Right to Own Property               |
| 18          | Freedom of Religion or Belief       |
| 19          | Freedom of Expression               |
| 20          | Freedom of Assembly                 |
| 21          | Right to Partake in Public Affairs  |
| 22          | Right to Social Security            |
| 23          | Right to Work                       |
| 24          | Right to Leisure and Rest           |
| 25          | Right to Adequate Living Standard   |
| 26          | Right to Education                  |
| 27          | Right to Partake in Cultural Life   |

Respond with **only** the article numbers in the following JSON format with the following fields:
- "primarily_limited": rights that the action directly and overwhelmingly infringes
- "substantially_limited": rights that the action infringes upon to a significant but not primary degree
- "moderately_limited": rights that the action infringes upon to a noticeable but less severe degree
- "somewhat_limited": rights the action impacts in a minor or peripheral way
- "somewhat_promoted": rights the action advances in a minor or peripheral way
- "moderately_promoted": rights the action advances to a noticeable but not overwhelming degree
- "substantially_promoted": rights the action advances to a significant but not primary degree
- "primarily_promoted": rights that the action directly and overwhelmingly advances"""
