import unittest
from src.generate_criteria import generate_acceptance_criteria

class TestGenerateCriteria(unittest.TestCase):
    def test_generate_criteria(self):
        description = "As a user, I want to reset my password so that I can regain access to my account if I forget it."
        criteria = generate_acceptance_criteria(description)
        self.assertIsInstance(criteria, str)
        self.assertGreater(len(criteria), 0)
        print("Test passed: Acceptance criteria generated successfully.")

if __name__ == "__main__":
    unittest.main()