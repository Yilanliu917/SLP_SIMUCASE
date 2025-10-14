"""Test improved disorder matching"""
from app.config import DISORDER_TYPES

# Test cases
test_cases = [
    'Child apraxia of speech',
    'child apraxia',
    'Childhood Apraxia of Speech',
    'expressive language disorders',
    'receptive language disorders',
    'pragmatics'
]

print('Testing disorder matching:')
print('='*60)

for test in test_cases:
    disorder_lower = test.lower()
    matched = False

    # Sort by length
    sorted_disorder_types = sorted(DISORDER_TYPES, key=len, reverse=True)

    for known_disorder in sorted_disorder_types:
        known_lower = known_disorder.lower()

        # Bidirectional check
        if known_lower in disorder_lower or disorder_lower in known_lower:
            print(f'[OK] "{test}" -> "{known_disorder}"')
            matched = True
            break

        # Apraxia special case
        disorder_words = set(disorder_lower.split())
        known_words = set(known_lower.split())

        if 'apraxia' in disorder_words and 'apraxia' in known_words:
            print(f'[OK] "{test}" -> "{known_disorder}" (apraxia match)')
            matched = True
            break

    if not matched:
        print(f'[FAIL] "{test}" -> No match found')

print('='*60)
