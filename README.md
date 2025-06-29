Quicksort Algorithm: Implementation, Analysis, and Randomization
Project Overview
This project provides a comprehensive implementation and analysis of the Quicksort algorithm, including both deterministic and randomized versions. The implementation includes detailed performance analysis, empirical testing, and theoretical complexity examination.

Features

Deterministic Quicksort: Traditional implementation using last element as pivot
Randomized Quicksort: Enhanced version with random pivot selection
Robust Implementation: Handles recursion limits with automatic iterative fallback
Performance Analysis: Comprehensive timing and operation counting
Empirical Testing: Automated testing across various input types and sizes
Visualization: Performance comparison charts and graphs
Theoretical Analysis: Detailed complexity analysis and mathematical foundations
Comprehensive Test Suite: Extensive testing including worst-case scenarios

Implementation Details
Design Decisions

Lomuto Partition Scheme: Chosen for simplicity and clarity
Last Element Pivot: Standard approach for deterministic version
Random Pivot Selection: Simple uniform random selection
In-place Sorting: Minimizes memory usage
Recursive Implementation: Clear and intuitive structure

Code Quality Features

Comprehensive Documentation: Detailed docstrings and comments
Type Hints: Full type annotation for better code clarity
Modular Design: Separate methods for different functionalities
Error Handling: Input validation and edge case management
Performance Monitoring: Built-in operation counting and timing
Recursion Safety: Automatic fallback to iterative implementation
Memory Efficiency: Stack-optimized algorithms for large inputs

Installation and Setup

1. Clone the repository

2. Install dependencies:
    pip install -r requirements.txt

3. Run the test suite:
    python quicksort_test_script.py
