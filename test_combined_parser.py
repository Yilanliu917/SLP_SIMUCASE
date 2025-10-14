"""Test combined table + text parsing"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.generation import parse_table_request
from app.config import FREE_MODELS, PREMIUM_MODELS

# Simulate the combined parsing logic
def test_combined_parse(table_file, text_request):
    tasks = []
    breakdown = ""

    # Parse table if provided
    if table_file:
        table_tasks, table_breakdown = parse_table_request(table_file)
        if table_tasks:
            tasks.extend(table_tasks)
            breakdown += table_breakdown + "\n\n"

    # Parse text for model override
    if text_request.strip():
        model_to_use = None
        request_lower = text_request.lower()

        all_models = FREE_MODELS + PREMIUM_MODELS
        for model in all_models:
            if model.lower() in request_lower:
                model_to_use = model
                break

        if tasks and model_to_use:
            for task in tasks:
                task['model'] = model_to_use
            breakdown += f"**Model Override:** All cases will use **{model_to_use}**\n\n"

    return tasks, breakdown

# Test 1: Table only
print("="*60)
print("TEST 1: Table only (no text)")
print("="*60)
tasks, breakdown = test_combined_parse('sample_students.csv', '')
print(breakdown.replace('📊', '[TABLE]').replace('✓', '[OK]'))
print(f"\nFirst task model: {tasks[0]['model'] if tasks else 'N/A'}")

# Test 2: Table + Model override
print("\n" + "="*60)
print("TEST 2: Table + 'Use Gemini 2.5 Pro'")
print("="*60)
tasks, breakdown = test_combined_parse('sample_students.csv', 'Use Gemini 2.5 Pro')
print(breakdown.replace('📊', '[TABLE]').replace('✓', '[OK]'))
print(f"\nFirst task model: {tasks[0]['model'] if tasks else 'N/A'}")

# Test 3: Table + GPT-4o override
print("\n" + "="*60)
print("TEST 3: Table + 'Use GPT-4o for all cases'")
print("="*60)
tasks, breakdown = test_combined_parse('sample_students.csv', 'Use GPT-4o for all cases')
print(breakdown.replace('📊', '[TABLE]').replace('✓', '[OK]'))
print(f"\nFirst task model: {tasks[0]['model'] if tasks else 'N/A'}")

print("\n" + "="*60)
print("VERIFICATION: All 4 students should use the specified model")
print("="*60)
for i, task in enumerate(tasks, 1):
    print(f"Student {i} ({task['id']}): Model = {task['model']}")
