"""
Run this directly: python direct_test.py
No server needed. Tests the OpenAI call in isolation.
"""
import json, re, os

KEY = os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    KEY = input("Paste your sk-proj-... key: ").strip()

print(f"Key prefix: {KEY[:20]}...")
print("Testing sync OpenAI call...")

try:
    from openai import OpenAI
    client = OpenAI(api_key=KEY)
    
    prompt = '''You are classifying a JEE Main Physics question.

QUESTION:
A ball is thrown upward with velocity 20 m/s. Find the maximum height. g=10 m/s2.

OPTIONS:
(1) 20 m
(2) 40 m
(3) 10 m
(4) 5 m

CHAPTERS AND TOPICS for Physics (ONLY choose from this list):
1. Motion in a Straight Line — Kinematics Equations | Graphs of Motion | Relative Motion | Free Fall and Projectile 1D
2. Laws of Motion — Newtons Three Laws | Friction | Free Body Diagrams
3. Work Energy and Power — Work Done by a Force | Kinetic Energy | Potential Energy | Conservation of Energy

DIFFICULTY GUIDE:
- easy: direct formula application, single step, standard result
- medium: 2-3 steps, concept combination, moderate calculation
- hard: multi-concept, tricky insight needed

INSTRUCTIONS:
- Return ONLY a JSON object, no markdown, no explanation.

Format: {"chapter": "<exact chapter name>", "topic": "<exact topic name or empty string>", "difficulty": "easy|medium|hard"}'''

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    print(f"Raw response: {raw!r}")
    data = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE))
    print(f"Parsed: {data}")
    print("\n✅ OpenAI call works!")

except Exception as e:
    import traceback
    print(f"\n❌ FAILED:")
    traceback.print_exc()