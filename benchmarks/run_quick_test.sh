#!/bin/bash
# Quick validation script for benchmarks
# Tests all benchmarks on small scale to ensure they work

set -e  # Exit on error

echo "======================================================================"
echo "Myriad Benchmark Quick Validation"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[1/4] Testing throughput benchmark...${NC}"
python benchmarks/throughput.py --num-envs 100 --num-steps 10

echo ""
echo -e "${BLUE}[2/4] Testing memory profiling...${NC}"
python benchmarks/memory_profile.py --profile envs --env cartpole

echo ""
echo -e "${BLUE}[3/4] Testing Myriad-only comparison...${NC}"
python benchmarks/comparison.py --library myriad --num-envs 100 --steps-per-env 100

echo ""
echo -e "${BLUE}[4/4] Testing performance demo...${NC}"
python examples/10_performance_demo.py --num-envs 500 --num-steps 10

echo ""
echo "======================================================================"
echo -e "${GREEN}✅ All benchmarks passed quick validation!${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Run full benchmark suite:"
echo "     python benchmarks/throughput.py --full"
echo ""
echo "  2. Profile memory at scale:"
echo "     python benchmarks/memory_profile.py --profile both"
echo ""
echo "  3. Compare against other libraries:"
echo "     pip install gymnax gymnasium"
echo "     python benchmarks/comparison.py"
echo ""
echo "  4. Generate plots:"
echo "     pip install matplotlib"
echo "     python benchmarks/plot_results.py"
echo ""
