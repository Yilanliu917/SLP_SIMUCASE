"""Test script for table parser"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.generation import parse_table_request

# Test the parser
tasks, confirmation = parse_table_request('sample_students.csv')

# Print confirmation (avoid emoji encoding issues in Windows terminal)
print("="*60)
print("CONFIRMATION MESSAGE:")
print("="*60)
# Replace emojis with ASCII equivalents for Windows terminal
safe_confirmation = confirmation.replace('📊', '[TABLE]').replace('✓', '[OK]').replace('❌', '[ERROR]')
print(safe_confirmation)

print("\n" + "="*60)
print("PARSED TASKS:")
print("="*60)
for i, task in enumerate(tasks, 1):
    print(f"\nTask {i}:")
    print(f"  ID: {task['id']}")
    print(f"  Grade: {task['grade']}")
    print(f"  Disorders: {', '.join(task['disorders'])}")
    print(f"  Model: {task['model']}")
    print(f"  Count: {task['count']}")

print(f"\n{'='*60}")
print(f"Total tasks parsed: {len(tasks)}")
print("="*60)
