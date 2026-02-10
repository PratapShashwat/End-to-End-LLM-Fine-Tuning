import json
import random

# CONFIG
INPUT_FILE = "test_set_full_data.json"

try:
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Could not find {INPUT_FILE}")
    exit()

# Pick 2 Random Files
selected = random.sample(data, 2)

print("="*60)
print("      MANUAL BIAS CHECK (2 FILES)")
print("="*60)

for i, entry in enumerate(selected):
    print(f"\n📄 DOCUMENT {i+1} (ID: {entry['id']})")
    print("-" * 30)
    
    print(f"🛑 GENERATED SUMMARY:\n{entry['generated']}\n")
    print(f"✅ REFERENCE SUMMARY:\n{entry['reference']}\n")
    
    print("="*60)