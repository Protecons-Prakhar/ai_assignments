import pandas as pd
import json
import time
import sys
import os

sys.path.append(os.path.abspath(".."))

from llm import ask

df = pd.read_csv("claims.csv")

claims_list = df["claim"].tolist()

claims_text = "\n".join(
    [f"{i+1}. {claim}" for i, claim in enumerate(claims_list)]
)

prompt = f"""
You are a P&C insurance claim classifier.

For EACH claim, classify into:

1. claim_type:
   - auto
   - property
   - liability
   - other

2. tone:
   - calm
   - frustrated
   - urgent

3. legal_action:
   - yes
   - no

Rules:
- Return ONLY valid JSON
- Return a JSON ARRAY
- Keep the original claim text
- Detect tone carefully
- Use different tones where appropriate

Example format:

[
  {{
    "claim": "My car was hit yesterday.",
    "claim_type": "auto",
    "tone": "frustrated",
    "legal_action": "no"
  }}
]

Claims:
{claims_text}
"""

try:
    response = ask([
        {
            "role": "system",
            "content": "You are an insurance classification assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ])

    content = response.choices[0].message.content.strip()

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    print("\nRAW RESPONSE:")
    print(content)

    results = json.loads(content)

    with open("output.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nClassification complete!")
    print("Output saved to output.json")

except Exception as e:

    print("\nError occurred:")
    print(e)

    with open("output.json", "w") as f:
        json.dump(
            [{
                "error": str(e)
            }],
            f,
            indent=4
        )