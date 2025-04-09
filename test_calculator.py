import unittest
from tool_manager import CalculatorTool

class TestCalculatorTool(unittest.TestCase):

    def setUp(self):
        self.calc = CalculatorTool()

    def test_addition(self):
        response = self.calc.execute("2+3")
        self.assertTrue(response.success)
        self.assertEqual(response.content, "5")

    def test_complex_expression(self):
        response = self.calc.execute("10+(5 * 3)-2")
        self.assertTrue(response.success)
        self.assertEqual(response.content, "23")

    def test_invalid_expression(self):
        response = self.calc.execute("What is universe?")  
        self.assertFalse(response.success)
        self.assertIn("Error:", response.content)

    def test_unexpected_characters(self):
        response = self.calc.execute("2+3     ¶¢[]|{}}")  
        self.assertTrue(response.success)
        self.assertEqual(response.content, "5")

    def test_division_by_zero(self):
        response = self.calc.execute("10 / 0")
        self.assertFalse(response.success)
        self.assertIn("division by zero", response.content.lower())

if __name__ == "__main__":
    unittest.main()
