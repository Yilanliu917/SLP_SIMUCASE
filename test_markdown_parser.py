"""Test the markdown parser with the existing case file"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.utils import parse_markdown_case, search_existing_cases_in_folder

# Test parsing a single case
print("="*60)
print("TEST 1: Parse existing case file")
print("="*60)

with open('generated_case_files/S5_S10.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by --- to get individual cases
sections = content.split('\n---\n')
print(f"\nFound {len(sections)} sections in file\n")

# Parse first valid case
for i, section in enumerate(sections):
    if '## S' in section:  # Look for case headers
        print(f"\nParsing section {i+1}...")
        case_data = parse_markdown_case(section)
        if case_data:
            print(f"[OK] Successfully parsed: {case_data['case_id']} - {case_data['name']}")
            print(f"  Grade: {case_data['grade']}")
            print(f"  Age: {case_data['age']}")
            print(f"  Gender: {case_data['gender']}")
            print(f"  Disorders: {case_data['disorders']}")
            print(f"  Annual Goals: {len(case_data['annual_goals'])} goals")
            print(f"  Session Notes: {len(case_data['session_notes'])} notes")
            if case_data['session_notes']:
                print(f"  First note preview: {case_data['session_notes'][0][:100]}...")
            break
        else:
            print(f"[FAIL] Failed to parse section {i+1}")

# Test folder search
print("\n" + "="*60)
print("TEST 2: Search for Kindergarten students with disorders")
print("="*60)

grades = ["Kindergarten", "Kindergarten"]
disorders_list = [["Receptive Language Disorders"], ["Articulation Disorders"]]

print(f"\nSearching for:")
print(f"  Member 1: Kindergarten - Receptive Language Disorders")
print(f"  Member 2: Kindergarten - Articulation Disorders\n")

matches = search_existing_cases_in_folder(grades, disorders_list)

print(f"\nFound {len(matches)} matching cases:")
for match in matches:
    print(f"  - {match['case_id']}: {match['name']} (Member {match['member_index']+1})")
    print(f"    Grade: {match['grade']}, Disorders: {match['disorders']}")
    print(f"    File: {os.path.basename(match['filepath'])}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
