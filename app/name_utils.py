"""
Name generation utilities for creating unique, realistic anonymous names.
"""
import random
from typing import List, Optional


# Name pools by locale
NAME_POOLS = {
    "en": {
        "first": [
            "Emily", "Michael", "Sarah", "David", "Jessica", "James", "Ashley",
            "Daniel", "Sophia", "Matthew", "Emma", "Joshua", "Olivia", "Andrew",
            "Isabella", "Christopher", "Madison", "Ryan", "Ava", "Nathan",
            "Grace", "Tyler", "Lily", "Brandon", "Chloe", "Jacob", "Ella",
            "Ethan", "Hannah", "Lucas", "Abigail", "Alexander", "Mia", "Noah",
            "William", "Charlotte", "Benjamin", "Amelia", "Samuel", "Harper",
            "Joseph", "Evelyn", "Dylan", "Aria", "Owen", "Scarlett", "Caleb",
            "Victoria", "Mason", "Zoey", "Zachary", "Nora", "Liam", "Riley",
            "Elijah", "Leah", "Logan", "Hazel", "Jack", "Violet", "Henry",
            "Aurora", "Isaac", "Savannah", "Wyatt", "Brooklyn", "Sebastian",
            "Bella", "Carter", "Claire", "Julian", "Skylar", "Grayson", "Lucy",
            "Aaron", "Aaliyah", "Cameron", "Audrey", "Connor", "Maya", "Landon",
            "Naomi", "Hunter", "Kennedy", "Jonathan", "Madelyn", "Eli", "Adeline"
        ],
        "last": [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
            "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
            "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker",
            "Young", "Allen", "King", "Wright", "Scott", "Green", "Baker",
            "Adams", "Nelson", "Hill", "Ramirez", "Campbell", "Mitchell", "Roberts",
            "Carter", "Phillips", "Evans", "Turner", "Torres", "Parker", "Collins",
            "Edwards", "Stewart", "Morris", "Nguyen", "Murphy", "Rivera", "Cook",
            "Rogers", "Morgan", "Peterson", "Cooper", "Reed", "Bailey", "Bell",
            "Gomez", "Kelly", "Howard", "Ward", "Cox", "Diaz", "Richardson",
            "Wood", "Watson", "Brooks", "Bennett", "Gray", "James", "Reyes",
            "Cruz", "Hughes", "Price", "Myers", "Long", "Foster", "Sanders"
        ]
    },
    "es": {
        "first": [
            "Sofia", "Mateo", "Isabella", "Santiago", "Valentina", "Sebastian",
            "Camila", "Diego", "Valeria", "Lucas", "Emma", "Miguel", "Lucia",
            "Gabriel", "Martina", "Alejandro", "Elena", "Carlos", "Paula",
            "Antonio", "Maria", "Jorge", "Ana", "Fernando", "Carmen",
            "Manuel", "Daniela", "Rafael", "Mariana", "Andres", "Carolina",
            "Francisco", "Gabriela", "Ricardo", "Adriana", "Eduardo", "Julia",
            "Roberto", "Natalia", "Pablo", "Victoria", "Jose", "Andrea",
            "Daniel", "Laura", "Javier", "Monica", "Marco", "Beatriz",
            "Luis", "Rosa", "Pedro", "Alicia", "Sergio", "Teresa",
            "Raul", "Patricia", "Oscar", "Sandra", "Victor", "Liliana",
            "Rodrigo", "Silvia", "Alberto", "Veronica", "Emilio", "Claudia",
            "Hector", "Isabel", "Arturo", "Diana", "Ignacio", "Cristina"
        ],
        "last": [
            "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Perez", "Sanchez", "Ramirez", "Torres", "Flores", "Rivera",
            "Gomez", "Diaz", "Cruz", "Morales", "Reyes", "Gutierrez",
            "Alvarez", "Mendez", "Jimenez", "Ruiz", "Fernandez", "Vargas",
            "Castillo", "Romero", "Herrera", "Medina", "Aguilar", "Ortiz",
            "Delgado", "Castro", "Moreno", "Ramos", "Navarro", "Vasquez",
            "Munoz", "Pena", "Campos", "Cortes", "Mendoza", "Rojas",
            "Guerrero", "Pacheco", "Nunez", "Soto", "Dominguez", "Vega",
            "Silva", "Fuentes", "Contreras", "Espinoza", "Maldonado", "Salazar",
            "Leon", "Rios", "Sandoval", "Carrillo", "Miranda", "Calderon",
            "Velasquez", "Estrada", "Cabrera", "Acosta", "Guzman", "Figueroa"
        ]
    }
}


def generate_anonymous_name(locale_pool: List[str] = ["en", "es"]) -> str:
    """
    Generate a single anonymous name from a pool of locales.

    Args:
        locale_pool: List of locale codes to choose from (e.g., ["en", "es"])

    Returns:
        A title-cased full name (e.g., "Emily Rodriguez")
    """
    # Choose a random locale from the pool
    locale = random.choice(locale_pool) if locale_pool else "en"

    # Fallback to 'en' if locale not found
    if locale not in NAME_POOLS:
        locale = "en"

    # Get name pools for this locale
    first_names = NAME_POOLS[locale]["first"]
    last_names = NAME_POOLS[locale]["last"]

    # Generate name
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)

    # Title-case and trim
    full_name = f"{first_name.strip().title()} {last_name.strip().title()}"

    return full_name


def generate_unique_names(
    n: int,
    locale_pool: List[str] = ["en", "es"],
    seed: Optional[int] = None
) -> List[str]:
    """
    Generate n unique anonymous names.

    Args:
        n: Number of unique names to generate
        locale_pool: List of locale codes to choose from
        seed: Optional random seed for reproducibility

    Returns:
        List of n unique full names
    """
    # Initialize random generator with seed if provided
    if seed is not None:
        rng = random.Random(seed)
        # Temporarily override the global random functions
        original_choice = random.choice
        random.choice = rng.choice
    else:
        rng = None

    unique_names = set()
    max_attempts = n * 100  # Prevent infinite loops
    attempts = 0

    while len(unique_names) < n and attempts < max_attempts:
        name = generate_anonymous_name(locale_pool)

        # If collision occurs, regenerate
        if name not in unique_names:
            unique_names.add(name)

        attempts += 1

    # Restore original random.choice if we overrode it
    if rng is not None:
        random.choice = original_choice

    # Convert to list and return
    result = list(unique_names)

    # If we couldn't generate enough unique names, warn
    if len(result) < n:
        print(f"Warning: Could only generate {len(result)} unique names out of {n} requested")

    return result


def test_name_generation():
    """Test function to verify name generation works correctly."""
    print("Testing name generation...")
    print("\n1. Generate 5 random names:")
    names = generate_unique_names(5)
    for i, name in enumerate(names, 1):
        print(f"   {i}. {name}")

    print("\n2. Generate 5 names with seed (should be reproducible):")
    names_seeded_1 = generate_unique_names(5, seed=42)
    for i, name in enumerate(names_seeded_1, 1):
        print(f"   {i}. {name}")

    print("\n3. Generate 5 names with same seed (should match above):")
    names_seeded_2 = generate_unique_names(5, seed=42)
    for i, name in enumerate(names_seeded_2, 1):
        print(f"   {i}. {name}")

    print(f"\n4. Verify reproducibility: {names_seeded_1 == names_seeded_2}")
    print(f"\n5. Verify uniqueness: {len(set(names)) == len(names)}")


if __name__ == "__main__":
    test_name_generation()
