"""
ADVANCED Equation Extractor for Math Word Problems
Uses multiple strategies: Pattern matching, syntactic parsing, and equation inference
"""

import re
import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try to load spaCy model, fallback to regex if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    logger.warning("spaCy model not available. Using regex-only mode.")
    SPACY_AVAILABLE = False
    nlp = None

class AdvancedMathParser:
    """
    Advanced math problem parser using multiple strategies
    """
    
    def __init__(self):
        self.variables = set()
        self.equations = []
        self.problem_type = "unknown"
        
        # Mathematical patterns database
        self.patterns = {
            'age': [
                (r'(\w)\s+is\s+(\d+)\s+years?\s+older\s+than\s+(\w)', self._parse_age_difference),
                (r'(\w)\s+is\s+(\d+)\s+years?\s+younger\s+than\s+(\w)', self._parse_age_difference),
                (r'sum.*?ages?.*?(\d+)', self._parse_age_sum),
            ],
            'ratio': [
                (r'ratio.*?(\w)\s*to\s*(\w).*?(\d+):(\d+)', self._parse_ratio),
                (r'(\w)\s*:\s*(\w)\s*=\s*(\d+)\s*:\s*(\d+)', self._parse_ratio_direct),
            ],
            'percentage': [
                (r'(\d+)%\s*of\s*(\w+)', self._parse_percentage_of),
                (r'(\d+)%\s*(?:discount|off)', self._parse_discount),
                (r'increase.*?(\d+)%', self._parse_percentage_increase),
                (r'decrease.*?(\d+)%', self._parse_percentage_decrease),
            ],
            'geometry': [
                (r'area.*?rectangle', self._parse_rectangle_area),
                (r'area.*?circle', self._parse_circle_area),
                (r'perimeter.*?square', self._parse_square_perimeter),
                (r'perimeter.*?rectangle', self._parse_rectangle_perimeter),
                (r'volume.*?cube', self._parse_cube_volume),
            ],
            'algebra': [
                (r'([xyz])\s*[\+\-]\s*(\d+)\s*=\s*(\d+)', self._parse_simple_equation),
                (r'(\d+)([xyz])\s*[\+\-]\s*(\d+)([xyz])\s*=\s*(\d+)', self._parse_linear_equation),
                (r'solve.*?([xyz])', self._parse_solve_for),
            ],
            'motion': [
                (r'speed.*?distance.*?time', self._parse_motion_basic),
                (r'km/h.*?hours?', self._parse_speed_time),
                (r'mph.*?miles', self._parse_speed_distance),
            ],
            'mixture': [
                (r'mixture.*?(\d+)%.*?(\d+)%', self._parse_mixture_percent),
                (r'solution.*?(\d+)%', self._parse_solution_concentration),
            ],
            'work': [
                (r'work.*?together.*?hours?', self._parse_work_together),
                (r'complete.*?work.*?hours?', self._parse_work_rate),
            ]
        }
        
        # Mathematical constants and formulas
        self.formulas = {
            'distance': 'distance = speed * time',
            'area_rectangle': 'area = length * width', 
            'area_circle': 'area = pi * radius^2',
            'perimeter_square': 'perimeter = 4 * side',
            'perimeter_rectangle': 'perimeter = 2 * (length + width)',
            'volume_cube': 'volume = side^3',
            'percentage': 'part = (percentage/100) * whole',
            'profit': 'profit = selling_price - cost_price',
            'discount': 'discount_price = original_price * (1 - discount/100)'
        }
    
    def parse_problem(self, text: str) -> Dict:
        """
        Advanced parsing with multiple strategies
        """
        self.variables = set()
        self.equations = []
        
        text_lower = text.lower().strip()
        original_text = text
        
        # Strategy 1: Direct equation extraction
        self._extract_explicit_equations(original_text)
        
        # Strategy 2: Pattern-based extraction
        self._extract_using_patterns(text_lower, original_text)
        
        # Strategy 3: Semantic parsing (if spaCy available)
        if SPACY_AVAILABLE:
            self._extract_using_semantics(original_text)
        
        # Strategy 4: Formula-based inference
        self._infer_from_formulas(text_lower, original_text)
        
        # Strategy 5: Number relationship analysis
        self._analyze_number_relationships(text_lower, original_text)
        
        # Clean and validate equations
        cleaned_equations = self._clean_equations(self.equations)
        
        # Classify problem type
        self.problem_type = self._advanced_classification(text_lower, cleaned_equations)
        
        return {
            'equations': cleaned_equations,
            'variables': list(self.variables),
            'problem_type': self.problem_type,
            'confidence': self._calculate_confidence(cleaned_equations, text_lower)
        }
    
    def _extract_explicit_equations(self, text: str):
        """Extract explicitly stated equations"""
        # Pattern 1: Simple equations like "x = 5 + 3"
        explicit_patterns = [
            r'([xyz]\s*=\s*[^\.!?]+)',  # x = expression
            r'(\w+\s*=\s*[^\.!?]+)',    # variable = expression
            r'([^\.!?]*=\s*[^\.!?]+)',  # anything = anything
        ]
        
        for pattern in explicit_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                equation = match.group(1).strip()
                if self._validate_equation(equation):
                    self.equations.append(equation)
                    # Extract variables
                    self.variables.update(re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', equation.split('=')[0].strip()))
    
    def _extract_using_patterns(self, text_lower: str, original_text: str):
        """Use predefined patterns to extract equations"""
        numbers = self._extract_numbers(original_text)
        
        for category, patterns in self.patterns.items():
            for pattern, handler in patterns:
                matches = re.search(pattern, text_lower)
                if matches:
                    try:
                        equations = handler(matches, numbers, original_text)
                        if equations:
                            self.equations.extend(equations)
                            break  # Use first matching pattern per category
                    except Exception as e:
                        logger.debug(f"Pattern handler failed: {e}")
                        continue
    
    def _extract_using_semantics(self, text: str):
        """Use NLP to understand problem structure"""
        if not SPACY_AVAILABLE:
            return
            
        doc = nlp(text)
        
        # Look for mathematical relationships
        for sent in doc.sents:
            # Check for "is" relationships (A is B)
            for token in sent:
                if token.lemma_ == "be" and token.head.pos_ == "VERB":
                    subject = self._get_subject(token)
                    complement = self._get_complement(token)
                    
                    if subject and complement and any(char.isdigit() for char in str(complement)):
                        # This might be a mathematical relationship
                        equation = f"{subject} = {complement}"
                        if self._validate_equation(equation):
                            self.equations.append(equation)
    
    def _infer_from_formulas(self, text_lower: str, original_text: str):
        """Infer equations based on known formulas"""
        numbers = self._extract_numbers(original_text)
        
        # Check which formulas might apply
        for formula_name, formula in self.formulas.items():
            if any(keyword in text_lower for keyword in formula_name.split('_')):
                # Replace generic variables with actual numbers if available
                inferred_eq = formula
                if numbers:
                    # Try to match numbers to variables
                    inferred_eq = self._instantiate_formula(formula, numbers, text_lower)
                
                if inferred_eq and self._validate_equation(inferred_eq):
                    self.equations.append(inferred_eq)
    
    def _analyze_number_relationships(self, text_lower: str, original_text: str):
        """Analyze numerical relationships in the problem"""
        numbers = self._extract_numbers(original_text)
        
        if len(numbers) >= 2:
            # Look for sum relationships
            if 'sum' in text_lower or 'total' in text_lower:
                total_match = re.search(r'sum.*?(\d+)', text_lower)
                if total_match:
                    total = total_match.group(1)
                    if numbers and len(numbers) >= 2:
                        self.equations.append(f"{numbers[0]} + {numbers[1]} = {total}")
            
            # Look for product relationships
            if 'product' in text_lower or 'times' in text_lower:
                product_match = re.search(r'product.*?(\d+)', text_lower)
                if product_match:
                    product = product_match.group(1)
                    if numbers and len(numbers) >= 2:
                        self.equations.append(f"{numbers[0]} * {numbers[1]} = {product}")
            
            # Look for difference relationships
            if 'difference' in text_lower:
                diff_match = re.search(r'difference.*?(\d+)', text_lower)
                if diff_match:
                    difference = diff_match.group(1)
                    if numbers and len(numbers) >= 2:
                        self.equations.append(f"{numbers[0]} - {numbers[1]} = {difference}")
    
    # Pattern handlers

    def _parse_ratio_direct(self, match, numbers, original_text):
        """Parse direct ratio notation A:B = C:D"""
        a, b, num1, num2 = match.groups()
        self.variables.update([a, b])
        return [f"{a}/{b} = {num1}/{num2}", f"{a} = ({num1}/{num2}) * {b}"]

    def _parse_discount(self, match, numbers, original_text):
        """Parse discount percentage problems"""
        discount_pct = match.group(1)
        # Find original price
        price_match = re.search(r'\$?(\d+)', original_text)
        if price_match:
            price = price_match.group(1)
            return [f"discount_price = {price} * (1 - {discount_pct}/100)"]
        return [f"discount_price = original_price * (1 - {discount_pct}/100)"]

    def _parse_percentage_increase(self, match, numbers, original_text):
        """Parse percentage increase problems"""
        increase_pct = match.group(1)
        # Find original value
        value_match = re.search(r'\b(\d+)\b', original_text)
        if value_match:
            value = value_match.group(1)
            return [f"new_value = {value} * (1 + {increase_pct}/100)"]
        return [f"new_value = original_value * (1 + {increase_pct}/100)"]

    def _parse_percentage_decrease(self, match, numbers, original_text):
        """Parse percentage decrease problems"""
        decrease_pct = match.group(1)
        # Find original value
        value_match = re.search(r'\b(\d+)\b', original_text)
        if value_match:
            value = value_match.group(1)
            return [f"new_value = {value} * (1 - {decrease_pct}/100)"]
        return [f"new_value = original_value * (1 - {decrease_pct}/100)"]

    def _parse_rectangle_area(self, match, numbers, original_text):
        """Parse rectangle area problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"area = {nums[0]} * {nums[1]}"]
        return ["area = length * width"]

    def _parse_circle_area(self, match, numbers, original_text):
        """Parse circle area problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"area = 3.14 * {nums[0]} * {nums[0]}"]
        return ["area = pi * radius^2"]

    def _parse_square_perimeter(self, match, numbers, original_text):
        """Parse square perimeter problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"perimeter = 4 * {nums[0]}"]
        return ["perimeter = 4 * side"]

    def _parse_rectangle_perimeter(self, match, numbers, original_text):
        """Parse rectangle perimeter problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"perimeter = 2 * ({nums[0]} + {nums[1]})"]
        return ["perimeter = 2 * (length + width)"]

    def _parse_cube_volume(self, match, numbers, original_text):
        """Parse cube volume problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"volume = {nums[0]}^3"]
        return ["volume = side^3"]

    def _parse_solve_for(self, match, numbers, original_text):
        """Parse 'solve for x' type problems"""
        variable = match.group(1)
        self.variables.add(variable)
        # Look for equation in context
        eq_match = re.search(r'([^\.!?]*=\s*[^\.!?]+)', original_text)
        if eq_match:
            return [eq_match.group(1)]
        return [f"{variable} = ?"]

    def _parse_speed_time(self, match, numbers, original_text):
        """Parse speed-time problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"distance = {nums[0]} * {nums[1]}"]
        return ["distance = speed * time"]

    def _parse_speed_distance(self, match, numbers, original_text):
        """Parse speed-distance problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"time = {nums[1]} / {nums[0]}"]  # time = distance / speed
        return ["time = distance / speed"]

    def _parse_mixture_percent(self, match, numbers, original_text):
        """Parse mixture problems with percentages"""
        pct1, pct2 = match.groups()
        nums = self._extract_numbers(original_text)
        if len(nums) >= 1:
            total = nums[0]
            return [f"({pct1}/100)*x + ({pct2}/100)*y = ({pct1}/100)*{total}"]
        return [f"({pct1}/100)*x + ({pct2}/100)*y = final_concentration*(x+y)"]

    def _parse_solution_concentration(self, match, numbers, original_text):
        """Parse solution concentration problems"""
        concentration = match.group(1)
        return [f"amount_solute = ({concentration}/100) * total_solution"]

    def _parse_work_together(self, match, numbers, original_text):
        """Parse work together problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"1/{nums[0]} + 1/{nums[1]} = 1/total_time"]
        return ["1/time_a + 1/time_b = 1/total_time"]

    def _parse_work_rate(self, match, numbers, original_text):
        """Parse work rate problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"work_rate = 1/{nums[0]}"]
        return ["work_rate = 1/time"]
    def _parse_age_difference(self, match, numbers, original_text):
        a, diff, b = match.groups()
        self.variables.update([a, b])
        return [f"{a} = {b} + {diff}"]
        
    def _parse_age_sum(self, match, numbers, original_text):
        total = match.group(1)
        if len(self.variables) >= 2:
            vars_list = list(self.variables)
            return [f"{vars_list[0]} + {vars_list[1]} = {total}"]
        return []
    
    def _parse_ratio(self, match, numbers, original_text):
        a, b, num1, num2 = match.groups()
        self.variables.update([a, b])
        return [f"{a}/{b} = {num1}/{num2}", f"{a} = ({num1}/{num2}) * {b}"]
    
    def _parse_percentage_of(self, match, numbers, original_text):
        pct, var = match.groups()
        self.variables.add(var)
        # Look for the result value
        result_match = re.search(r'is\s*(\d+)', original_text.lower())
        if result_match:
            return [f"({pct}/100) * {var} = {result_match.group(1)}"]
        return [f"({pct}/100) * {var}"]
    
    def _parse_simple_equation(self, match, numbers, original_text):
        var, num1, num2 = match.groups()
        self.variables.add(var)
        return [f"{var} + {num1} = {num2}"]
    
    def _parse_linear_equation(self, match, numbers, original_text):
        coeff1, var1, coeff2, var2, result = match.groups()
        self.variables.update([var1, var2])
        return [f"{coeff1}{var1} + {coeff2}{var2} = {result}"]
    
    def _parse_motion_basic(self, match, numbers, original_text):
        if len(numbers) >= 3:
            return [f"{numbers[0]} = {numbers[1]} * {numbers[2]}"]  # d = s * t
        return ["distance = speed * time"]
    
    # Helper methods
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text"""
        return re.findall(r'\b(\d+)\b', text)
    
    def _validate_equation(self, equation: str) -> bool:
        """Validate if an equation is mathematically plausible"""
        if not equation or '=' not in equation:
            return False
        
        parts = equation.split('=')
        if len(parts) != 2:
            return False
        
        left, right = parts[0].strip(), parts[1].strip()
        
        # Both sides should contain mathematical content
        left_math = any(c in left for c in '+*-/^()') or any(word in left for word in ['sqrt', 'pi'])
        right_math = any(c in right for c in '+*-/^()') or any(word in right for word in ['sqrt', 'pi'])
        
        if not (left_math or right_math):
            return False
        
        return True
    
    def _clean_equations(self, equations: List[str]) -> List[str]:
        """Clean and normalize equations"""
        cleaned = []
        for eq in equations:
            # Normalize whitespace
            eq = re.sub(r'\s+', ' ', eq.strip())
            # Remove trailing punctuation
            eq = re.sub(r'[\.!?,;:]$', '', eq)
            # Replace × with *
            eq = eq.replace('×', '*')
            # Replace ÷ with /
            eq = eq.replace('÷', '/')
            
            if self._validate_equation(eq):
                cleaned.append(eq)
        
        return list(set(cleaned))  # Remove duplicates
    
    def _advanced_classification(self, text_lower: str, equations: List[str]) -> str:
        """Advanced problem type classification"""
        scores = defaultdict(int)
        
        # Keyword-based scoring
        keyword_mapping = {
            'age': ['year', 'age', 'old', 'young'],
            'ratio': ['ratio', 'proportion', ':'],
            'percentage': ['%', 'percent'],
            'geometry': ['area', 'perimeter', 'volume', 'circle', 'rectangle', 'square'],
            'algebra': ['solve', 'equation', 'variable', 'x', 'y', 'z'],
            'motion': ['speed', 'distance', 'time', 'km/h', 'mph'],
            'mixture': ['mixture', 'solution', 'concentration'],
            'work': ['work', 'complete', 'together', 'rate']
        }
        
        for category, keywords in keyword_mapping.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        # Equation-based scoring
        for eq in equations:
            if 'age' in eq or 'year' in eq:
                scores['age'] += 2
            if '/' in eq and '=' in eq and any(c in eq for c in 'xyz'):
                scores['ratio'] += 2
            if '%' in eq or '100' in eq:
                scores['percentage'] += 2
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'unknown'
    
    def _calculate_confidence(self, equations: List[str], text_lower: str) -> float:
        """Calculate confidence score for the extraction"""
        if not equations:
            return 0.0
        
        confidence = 0.0
        confidence += min(len(equations) * 0.2, 0.6)  # More equations = more confidence
        
        # Check if equations contain variables mentioned in text
        for eq in equations:
            if any(var in text_lower for var in self.variables):
                confidence += 0.2
        
        # Check for mathematical operations
        for eq in equations:
            if any(op in eq for op in ['+', '-', '*', '/']):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_subject(self, token):
        """Get subject of a verb (simplified)"""
        for child in token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child.text
        return None
    
    def _get_complement(self, token):
        """Get complement of a verb (simplified)"""
        for child in token.children:
            if child.dep_ in ["attr", "acomp", "dobj"]:
                return child.text
        return None
    
    def _instantiate_formula(self, formula: str, numbers: List[str], context: str) -> str:
        """Try to instantiate a generic formula with actual numbers"""
        # This is a simplified version - in production you'd want more sophisticated matching
        if len(numbers) >= 2:
            return formula.replace('length', numbers[0]).replace('width', numbers[1])
        return formula

def extract_equations_advanced(text: str) -> List[str]:
    """
    Main advanced extraction function
    """
    parser = AdvancedMathParser()
    result = parser.parse_problem(text)
    
    logger.info(f"Extracted {len(result['equations'])} equations with confidence {result['confidence']:.2f}")
    
    return result['equations']

# Test function
def test_advanced_parser():
    """Test the advanced parser"""
    test_cases = [
        "A is 5 years older than B. The sum of their ages is 25.",
        "The ratio of boys to girls is 3:2. There are 15 boys.",
        "25% of 200 is what number?",
        "A rectangle has length 10 and width 5. What is its area?",
        "x + 5 = 20",
        "If a car travels at 60 km/h for 2 hours, how far does it go?",
        "The sum of two numbers is 40 and their difference is 10.",
        "A store offers 20% discount on a $50 item.",
        "The product of two numbers is 48 and their sum is 14."
    ]
    
    parser = AdvancedMathParser()
    for problem in test_cases:
        print(f"\n{'='*50}")
        print(f"Problem: {problem}")
        result = parser.parse_problem(problem)
        print(f"Equations: {result['equations']}")
        print(f"Variables: {result['variables']}")
        print(f"Type: {result['problem_type']}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    test_advanced_parser()