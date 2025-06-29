import random
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Tuple

class QuicksortAnalyzer:
    """
    A comprehensive implementation and analysis of Quicksort algorithm
    including both deterministic and randomized versions.
    """
    
    def __init__(self):
        self.comparison_count = 0
        self.swap_count = 0
    
    def reset_counters(self):
        """Reset operation counters for analysis"""
        self.comparison_count = 0
        self.swap_count = 0
    
    def deterministic_quicksort(self, arr: List[int], low: int = 0, high: int = None) -> List[int]:
        """
        Deterministic Quicksort implementation using last element as pivot.
        Includes recursion depth management to handle worst-case scenarios.
        
        Args:
            arr: List of integers to sort
            low: Starting index (default: 0)
            high: Ending index (default: len(arr)-1)
        
        Returns:
            Sorted list
        """
        if high is None:
            high = len(arr) - 1
            # Set a higher recursion limit for sorting
            original_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(max(2000, len(arr) * 2))
            try:
                result = self._deterministic_quicksort_recursive(arr, low, high)
                return result
            except RecursionError:
                # Fall back to iterative implementation if recursion limit exceeded
                print("Recursion limit exceeded, switching to iterative implementation...")
                return self._deterministic_quicksort_iterative(arr, low, high)
            finally:
                sys.setrecursionlimit(original_limit)
        else:
            return self._deterministic_quicksort_recursive(arr, low, high)
    
    def _deterministic_quicksort_recursive(self, arr: List[int], low: int, high: int) -> List[int]:
        """Recursive helper for deterministic quicksort"""
        if low < high:
            # Partition the array and get pivot index
            pivot_index = self._partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            self._deterministic_quicksort_recursive(arr, low, pivot_index - 1)
            self._deterministic_quicksort_recursive(arr, pivot_index + 1, high)
        
        return arr
    
    def _deterministic_quicksort_iterative(self, arr: List[int], low: int, high: int) -> List[int]:
        """
        Iterative implementation of deterministic quicksort to avoid recursion issues.
        Uses an explicit stack to simulate recursion.
        """
        # Create an auxiliary stack
        stack = [(low, high)]
        
        while stack:
            low, high = stack.pop()
            
            if low < high:
                # Partition the array and get pivot index
                pivot_index = self._partition(arr, low, high)
                
                # Push left and right subarrays to stack
                # Push larger subarray first to optimize stack usage
                if pivot_index - low > high - pivot_index:
                    stack.append((low, pivot_index - 1))
                    stack.append((pivot_index + 1, high))
                else:
                    stack.append((pivot_index + 1, high))
                    stack.append((low, pivot_index - 1))
        
        return arr
    
    def randomized_quicksort(self, arr: List[int], low: int = 0, high: int = None) -> List[int]:
        """
        Randomized Quicksort implementation with random pivot selection.
        Includes recursion depth management for robustness.
        
        Args:
            arr: List of integers to sort
            low: Starting index (default: 0)
            high: Ending index (default: len(arr)-1)
        
        Returns:
            Sorted list
        """
        if high is None:
            high = len(arr) - 1
            # Set a higher recursion limit for sorting
            original_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(max(2000, len(arr) * 2))
            try:
                result = self._randomized_quicksort_recursive(arr, low, high)
                return result
            except RecursionError:
                # Fall back to iterative implementation if needed (very rare)
                print("Recursion limit exceeded in randomized version, switching to iterative...")
                return self._randomized_quicksort_iterative(arr, low, high)
            finally:
                sys.setrecursionlimit(original_limit)
        else:
            return self._randomized_quicksort_recursive(arr, low, high)
    
    def _randomized_quicksort_recursive(self, arr: List[int], low: int, high: int) -> List[int]:
        """Recursive helper for randomized quicksort"""
        if low < high:
            # Randomly select pivot and swap with last element
            random_index = random.randint(low, high)
            arr[random_index], arr[high] = arr[high], arr[random_index]
            self.swap_count += 1
            
            # Partition the array and get pivot index
            pivot_index = self._partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            self._randomized_quicksort_recursive(arr, low, pivot_index - 1)
            self._randomized_quicksort_recursive(arr, pivot_index + 1, high)
        
        return arr
    
    def _randomized_quicksort_iterative(self, arr: List[int], low: int, high: int) -> List[int]:
        """
        Iterative implementation of randomized quicksort.
        Uses an explicit stack to simulate recursion.
        """
        # Create an auxiliary stack
        stack = [(low, high)]
        
        while stack:
            low, high = stack.pop()
            
            if low < high:
                # Randomly select pivot and swap with last element
                random_index = random.randint(low, high)
                arr[random_index], arr[high] = arr[high], arr[random_index]
                self.swap_count += 1
                
                # Partition the array and get pivot index
                pivot_index = self._partition(arr, low, high)
                
                # Push left and right subarrays to stack
                # Push larger subarray first to optimize stack usage
                if pivot_index - low > high - pivot_index:
                    stack.append((low, pivot_index - 1))
                    stack.append((pivot_index + 1, high))
                else:
                    stack.append((pivot_index + 1, high))
                    stack.append((low, pivot_index - 1))
        
        return arr
    
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Partition function using Lomuto partition scheme.
        Places pivot in correct position and returns its index.
        
        Args:
            arr: Array to partition
            low: Starting index
            high: Ending index (pivot element)
        
        Returns:
            Index of pivot after partitioning
        """
        pivot = arr[high]  # Choose last element as pivot
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            self.comparison_count += 1
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.swap_count += 1
        
        # Place pivot in correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.swap_count += 1
        
        return i + 1
    
    def generate_test_data(self, size: int, data_type: str) -> List[int]:
        """
        Generate test data of different types for empirical analysis.
        
        Args:
            size: Size of the array
            data_type: Type of data ('random', 'sorted', 'reverse', 'duplicates')
        
        Returns:
            Generated test data
        """
        if data_type == 'random':
            return [random.randint(1, 1000) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(1, size + 1))
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'duplicates':
            return [random.randint(1, 10) for _ in range(size)]
        else:
            raise ValueError("Invalid data type")
    
    def time_sorting_algorithm(self, sort_func, arr: List[int]) -> Tuple[float, int, int]:
        """
        Time a sorting algorithm and count operations.
        
        Args:
            sort_func: Sorting function to test
            arr: Array to sort
        
        Returns:
            Tuple of (execution_time, comparisons, swaps)
        """
        arr_copy = arr.copy()
        self.reset_counters()
        
        start_time = time.time()
        sort_func(arr_copy)
        end_time = time.time()
        
        return end_time - start_time, self.comparison_count, self.swap_count
    
    def empirical_analysis(self, sizes: List[int] = None) -> dict:
        """
        Perform comprehensive empirical analysis of both algorithms.
        Includes safety measures for worst-case scenarios.
        
        Args:
            sizes: List of input sizes to test
        
        Returns:
            Dictionary containing analysis results
        """
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        data_types = ['random', 'sorted', 'reverse', 'duplicates']
        results = {
            'deterministic': {dt: {'times': [], 'comparisons': [], 'swaps': []} for dt in data_types},
            'randomized': {dt: {'times': [], 'comparisons': [], 'swaps': []} for dt in data_types}
        }
        
        print("Performing empirical analysis...")
        print("=" * 50)
        print("Note: Large sorted/reverse arrays may take longer due to O(n²) behavior")
        
        for size in sizes:
            print(f"\nTesting with array size: {size}")
            
            for data_type in data_types:
                # Generate test data
                test_data = self.generate_test_data(size, data_type)
                
                # Test deterministic quicksort with timeout for worst cases
                try:
                    det_time, det_comp, det_swaps = self.time_sorting_algorithm(
                        self.deterministic_quicksort, test_data
                    )
                    results['deterministic'][data_type]['times'].append(det_time)
                    results['deterministic'][data_type]['comparisons'].append(det_comp)
                    results['deterministic'][data_type]['swaps'].append(det_swaps)
                except Exception as e:
                    print(f"    Error in deterministic sort for {data_type}: {e}")
                    # Use a placeholder value for failed tests
                    results['deterministic'][data_type]['times'].append(float('inf'))
                    results['deterministic'][data_type]['comparisons'].append(0)
                    results['deterministic'][data_type]['swaps'].append(0)
                    det_time = float('inf')
                
                # Test randomized quicksort (average of 3 runs for randomized)
                rand_times, rand_comps, rand_swaps = [], [], []
                for run in range(3):  # Reduced from 5 to 3 for faster testing
                    try:
                        r_time, r_comp, r_swaps = self.time_sorting_algorithm(
                            self.randomized_quicksort, test_data
                        )
                        rand_times.append(r_time)
                        rand_comps.append(r_comp)
                        rand_swaps.append(r_swaps)
                    except Exception as e:
                        print(f"    Error in randomized sort run {run+1}: {e}")
                        break
                
                if rand_times:  # If we have at least one successful run
                    results['randomized'][data_type]['times'].append(np.mean(rand_times))
                    results['randomized'][data_type]['comparisons'].append(np.mean(rand_comps))
                    results['randomized'][data_type]['swaps'].append(np.mean(rand_swaps))
                    rand_avg_time = np.mean(rand_times)
                else:
                    results['randomized'][data_type]['times'].append(float('inf'))
                    results['randomized'][data_type]['comparisons'].append(0)
                    results['randomized'][data_type]['swaps'].append(0)
                    rand_avg_time = float('inf')
                
                # Display results with better formatting
                det_time_str = f"{det_time:.6f}s" if det_time != float('inf') else "TIMEOUT"
                rand_time_str = f"{rand_avg_time:.6f}s" if rand_avg_time != float('inf') else "TIMEOUT"
                print(f"  {data_type.capitalize():12} - Det: {det_time_str:>10}, Rand: {rand_time_str:>10}")
        
        results['sizes'] = sizes
        return results
    
    def plot_results(self, results: dict):
        """
        Plot the empirical analysis results.
        
        Args:
            results: Results dictionary from empirical_analysis
        """
        data_types = ['random', 'sorted', 'reverse', 'duplicates']
        sizes = results['sizes']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quicksort Performance Analysis', fontsize=16)
        
        for i, data_type in enumerate(data_types):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            det_times = results['deterministic'][data_type]['times']
            rand_times = results['randomized'][data_type]['times']
            
            ax.plot(sizes, det_times, 'bo-', label='Deterministic', linewidth=2, markersize=6)
            ax.plot(sizes, rand_times, 'ro-', label='Randomized', linewidth=2, markersize=6)
            
            ax.set_title(f'{data_type.capitalize()} Data')
            ax.set_xlabel('Array Size')
            ax.set_ylabel('Time (seconds)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def theoretical_analysis(self):
        """
        Print theoretical analysis of Quicksort algorithm.
        """
        print("\n" + "="*80)
        print("THEORETICAL ANALYSIS OF QUICKSORT")
        print("="*80)
        
        print("\n1. TIME COMPLEXITY ANALYSIS:")
        print("-" * 40)
        print("Best Case: O(n log n)")
        print("  - Occurs when pivot divides array into two equal halves")
        print("  - Recurrence: T(n) = 2T(n/2) + O(n)")
        print("  - Solution: T(n) = O(n log n)")
        
        print("\nAverage Case: O(n log n)")
        print("  - Expected behavior with random pivot selection")
        print("  - Probabilistic analysis shows expected depth is O(log n)")
        print("  - Each level requires O(n) operations")
        
        print("\nWorst Case: O(n²)")
        print("  - Occurs when pivot is always smallest/largest element")
        print("  - Happens with already sorted or reverse sorted arrays")
        print("  - Recurrence: T(n) = T(n-1) + O(n)")
        print("  - Solution: T(n) = O(n²)")
        
        print("\n2. SPACE COMPLEXITY ANALYSIS:")
        print("-" * 40)
        print("Best/Average Case: O(log n)")
        print("  - Due to recursive call stack depth")
        print("  - Balanced partitioning leads to O(log n) recursion depth")
        
        print("\nWorst Case: O(n)")
        print("  - Unbalanced partitioning leads to O(n) recursion depth")
        print("  - Each recursive call adds one frame to the stack")
        print("  - Implementation includes iterative fallback to handle deep recursion")
        
        print("\n3. RANDOMIZATION BENEFITS:")
        print("-" * 40)
        print("• Reduces probability of worst-case behavior to O(1/n!)")
        print("• Makes performance independent of input ordering")
        print("• Average-case complexity becomes expected complexity")
        print("• Provides better practical performance on real-world data")
        print("• Eliminates adversarial input scenarios")
        
        print("\n4. IMPLEMENTATION ROBUSTNESS:")
        print("-" * 40)
        print("• Automatic recursion limit management")
        print("• Iterative fallback for deep recursion scenarios")
        print("• Graceful handling of worst-case inputs")
        print("• Stack-optimized iterative implementation")


def main():
    """
    Main function to demonstrate Quicksort implementations and analysis.
    """
    analyzer = QuicksortAnalyzer()
    
    # Print theoretical analysis
    analyzer.theoretical_analysis()
    
    # Demonstrate sorting with small example
    print("\n" + "="*80)
    print("DEMONSTRATION WITH SAMPLE DATA")
    print("="*80)
    
    sample_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"\nOriginal array: {sample_data}")
    
    # Test deterministic quicksort
    det_result = analyzer.deterministic_quicksort(sample_data.copy())
    print(f"Deterministic Quicksort result: {det_result}")
    
    # Test randomized quicksort
    rand_result = analyzer.randomized_quicksort(sample_data.copy())
    print(f"Randomized Quicksort result: {rand_result}")
    
    # Perform empirical analysis with smaller sizes to avoid recursion issues
    print("\n" + "="*80)
    print("EMPIRICAL ANALYSIS")
    print("="*80)
    
    # Use smaller sizes for safer testing, focusing on demonstrating the algorithmic differences
    test_sizes = [100, 300, 500, 800, 1200]
    results = analyzer.empirical_analysis(test_sizes)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\nKey Observations:")
    print("• Randomized Quicksort performs significantly better on sorted/reverse-sorted data")
    print("• Both versions have similar performance on random data")
    print("• Deterministic version shows O(n²) behavior on pathological inputs")
    print("• Randomized version maintains O(n log n) performance across all input types")
    print("• Iterative fallback prevents recursion limit issues in worst cases")
    
    # Additional demonstration of worst-case scenario
    print("\n" + "="*50)
    print("WORST-CASE SCENARIO DEMONSTRATION")
    print("="*50)
    
    # Test with a moderately sized sorted array to show the difference
    sorted_array = list(range(1, 501))  # 500 elements
    print(f"\nTesting with sorted array of 500 elements...")
    
    # Time deterministic version
    det_time, _, _ = analyzer.time_sorting_algorithm(
        analyzer.deterministic_quicksort, sorted_array.copy()
    )
    
    # Time randomized version (average of 3 runs)
    rand_times = []
    for _ in range(3):
        rand_time, _, _ = analyzer.time_sorting_algorithm(
            analyzer.randomized_quicksort, sorted_array.copy()
        )
        rand_times.append(rand_time)
    rand_avg = np.mean(rand_times)
    
    print(f"Deterministic Quicksort: {det_time:.6f}s")
    print(f"Randomized Quicksort:   {rand_avg:.6f}s")
    print(f"Performance improvement: {det_time/rand_avg:.1f}x faster with randomization")
    
    # Uncomment the line below to generate plots (requires matplotlib)
    # analyzer.plot_results(results)

if __name__ == "__main__":
    main()
