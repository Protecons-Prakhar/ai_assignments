import pandas as pd
import json
import time

import sys
import os

sys.path.append(os.path.abspath(".."))

from llm import ask

# Read claims CSV
df = pd.read_csv("claims.csv")

results = []

for index, row in df.iterrows():

    claim = row["claim"]

    prompt = f"""
You are a P&C insurance claim classifier.

Classify this claim into:

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

Return ONLY valid JSON.

Example:
{{
  "claim_type": "auto",
  "tone": "calm",
  "legal_action": "no"
}}

Claim:
{claim}
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

        # Remove markdown if model returns ```json
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        print("\nRAW RESPONSE:")
        print(content)

        parsed = json.loads(content)

        parsed["claim"] = claim

        results.append(parsed)

        print(f"\nProcessed claim {index + 1}")

        # IMPORTANT:
        # Gemini free tier rate limit fix
        time.sleep(12)

    except Exception as e:

        print(f"\nError processing claim {index + 1}:")
        print(e)

        # Add fallback object so all 10 claims exist
        results.append({
            "claim": claim,
            "claim_type": "other",
            "tone": "calm",
            "legal_action": "no",
            "error": str(e)
        })

# Save JSON output
with open("output.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nClassification complete!")
print("Output saved to output.json")