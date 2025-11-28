from typing import List, Optional, Tuple

# (prompt, gold answer) aligned with math-style prompts used in rldynamic math500
PROBES: List[Tuple[str, Optional[str]]] = [
    ("Let's think step by step and output the final answer in \\boxed{your answer here}. What is 12 + 15?", "27"),
    ("Solve and box the answer: If x = 7, what is x^2 + 5x + 6?", "90"),
    ("Compute the product and box it: What is 14 * 6?", "84"),
    ("Find the value and box it: What is the greatest common divisor of 18 and 30?", "6"),
    ("Solve the equation for x and box the answer: 3x - 9 = 0", "3"),
]
