#!/bin/bash

#
# Matthew Abbott 2025
# GNN Tests - Testing gnn.cu and facaded_gnn.cu
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output_gnn_cuda"
GNN_BIN="./gnn_cuda"
FACADE_BIN="./facaded_gnn_cuda"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup/Cleanup
cleanup() {
    :
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR"

# Compile both GNN implementations
echo -e "${BLUE}Compiling GNN implementations...${NC}"
echo ""

echo "Compiling gnn.cu..."
nvcc -o gnn_cuda gnn.cu 2>&1 | grep -v "pragma message" || true

echo "Compiling facaded_gnn.cu..."
nvcc -o facaded_gnn_cuda facaded_gnn.cu 2>&1 | grep -v "pragma message" || true

echo ""

# Test function
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        echo "$output" | head -5
        FAIL=$((FAIL + 1))
    fi
}

run_test_exit_code() {
    local test_name="$1"
    local command="$2"
    local expected_code="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if [ $exit_code -eq $expected_code ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected exit code: $expected_code, got: $exit_code"
        FAIL=$((FAIL + 1))
    fi
}

check_file_exists() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
    fi
}

check_file_size_nonzero() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file" ] && [ -s "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File not found or empty: $file"
        FAIL=$((FAIL + 1))
    fi
}

compare_files() {
    local test_name="$1"
    local file1="$2"
    local file2="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file1" ] && [ -f "$file2" ]; then
        if cmp -s "$file1" "$file2"; then
            echo -e "${GREEN}PASS${NC}"
            PASS=$((PASS + 1))
        else
            echo -e "${RED}FAIL${NC}"
            echo "  Files differ: $file1 vs $file2"
            FAIL=$((FAIL + 1))
        fi
    else
        echo -e "${RED}FAIL${NC}"
        echo "  One or both files missing"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
# Start Tests
# ============================================

echo ""
echo "========================================="
echo "GNN CUDA Comprehensive Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$GNN_BIN" ]; then
    echo -e "${RED}Error: $GNN_BIN not found. Compile failed.${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile failed.${NC}"
    exit 1
fi

echo -e "${BLUE}=== GNN CUDA Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "GNN help command" \
    "$GNN_BIN help" \
    "COMMANDS:"

run_test \
    "GNN --help flag" \
    "$GNN_BIN --help" \
    "COMMANDS:"

run_test \
    "GNN -h flag" \
    "$GNN_BIN -h" \
    "COMMANDS:"

run_test \
    "GNN no args shows help" \
    "$GNN_BIN" \
    "USAGE:"

run_test \
    "Facade GNN help command" \
    "$FACADE_BIN help" \
    "COMMANDS:"

run_test \
    "Facade GNN --help flag" \
    "$FACADE_BIN --help" \
    "COMMANDS:"

run_test \
    "GNN help shows create command" \
    "$GNN_BIN help" \
    "create"

run_test \
    "GNN help shows train command" \
    "$GNN_BIN help" \
    "train"

run_test \
    "GNN help shows predict command" \
    "$GNN_BIN help" \
    "predict"

run_test \
    "GNN help shows pagerank command" \
    "$GNN_BIN help" \
    "pagerank"

run_test \
    "GNN help shows degree command" \
    "$GNN_BIN help" \
    "degree"

run_test \
    "GNN help shows neighbors command" \
    "$GNN_BIN help" \
    "neighbors"

run_test \
    "GNN help shows gradient-flow command" \
    "$GNN_BIN help" \
    "gradient-flow"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create basic GNN model" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/basic_gnn.bin" \
    "Created GNN model"

check_file_exists \
    "Model file created" \
    "$TEMP_DIR/basic_gnn.bin"

check_file_size_nonzero \
    "Model file non-empty" \
    "$TEMP_DIR/basic_gnn.bin"

run_test \
    "Create GNN with custom architecture" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=5 --mp-layers=4 --save=$TEMP_DIR/custom_gnn.bin" \
    "Feature size: 10"

run_test \
    "Create shows hidden size" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=5 --mp-layers=4 --save=$TEMP_DIR/arch1.bin" \
    "Hidden size: 64"

run_test \
    "Create shows output size" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=5 --mp-layers=4 --save=$TEMP_DIR/arch2.bin" \
    "Output size: 5"

run_test \
    "Create shows MP layers" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=5 --mp-layers=4 --save=$TEMP_DIR/arch3.bin" \
    "Message passing layers: 4"

run_test \
    "Create minimal model" \
    "$GNN_BIN create --feature=1 --hidden=1 --output=1 --mp-layers=1 --save=$TEMP_DIR/minimal.bin" \
    "Created GNN model"

echo ""

# ============================================
# Model Creation - Hyperparameters
# ============================================

echo -e "${BLUE}Group: Model Creation - Hyperparameters${NC}"

run_test \
    "Create with custom learning rate" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --lr=0.001 --save=$TEMP_DIR/lr_test.bin" \
    "Learning rate: 0.0010"

run_test \
    "Create with ReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=relu --save=$TEMP_DIR/relu_test.bin" \
    "Activation: relu"

run_test \
    "Create with LeakyReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=leakyrelu --save=$TEMP_DIR/leaky_test.bin" \
    "Activation: leakyrelu"

run_test \
    "Create with Tanh activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=tanh --save=$TEMP_DIR/tanh_test.bin" \
    "Activation: tanh"

run_test \
    "Create with Sigmoid activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=sigmoid --save=$TEMP_DIR/sigmoid_test.bin" \
    "Activation: sigmoid"

run_test \
    "Create with MSE loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --loss=mse --save=$TEMP_DIR/mse_test.bin" \
    "Loss function: mse"

run_test \
    "Create with BCE loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --loss=bce --save=$TEMP_DIR/bce_test.bin" \
    "Loss function: bce"

run_test \
    "Create with all hyperparameters" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=3 --lr=0.005 --activation=leakyrelu --loss=bce --save=$TEMP_DIR/all_hyper.bin" \
    "Created GNN model"

echo ""

# ============================================
# Model Info
# ============================================

echo -e "${BLUE}Group: Model Info${NC}"

run_test \
    "Info shows feature size" \
    "$GNN_BIN info --model=$TEMP_DIR/basic_gnn.bin" \
    "Feature size: 3"

run_test \
    "Info shows hidden size" \
    "$GNN_BIN info --model=$TEMP_DIR/basic_gnn.bin" \
    "Hidden size: 16"

run_test \
    "Info shows output size" \
    "$GNN_BIN info --model=$TEMP_DIR/basic_gnn.bin" \
    "Output size: 2"

run_test \
    "Info shows MP layers" \
    "$GNN_BIN info --model=$TEMP_DIR/basic_gnn.bin" \
    "Message passing layers: 2"

run_test \
    "Info header present" \
    "$GNN_BIN info --model=$TEMP_DIR/basic_gnn.bin" \
    "GNN Model Information"

run_test \
    "Info shows GPU enabled" \
    "$GNN_BIN info --model=$TEMP_DIR/basic_gnn.bin" \
    "GPU Acceleration: Enabled"

echo ""

# ============================================
# Save and Load
# ============================================

echo -e "${BLUE}Group: Save and Load${NC}"

run_test \
    "Load model" \
    "$GNN_BIN load --model=$TEMP_DIR/basic_gnn.bin" \
    "Model loaded"

run_test \
    "Save model to new file" \
    "$GNN_BIN save --model=$TEMP_DIR/basic_gnn.bin --output=$TEMP_DIR/saved_copy.bin" \
    "Model saved"

check_file_exists \
    "Saved copy exists" \
    "$TEMP_DIR/saved_copy.bin"

compare_files \
    "Saved copy matches original" \
    "$TEMP_DIR/basic_gnn.bin" \
    "$TEMP_DIR/saved_copy.bin"

run_test \
    "Load and verify architecture" \
    "$GNN_BIN info --model=$TEMP_DIR/saved_copy.bin" \
    "Feature size: 3"

echo ""

# ============================================
# Graph Operations
# ============================================

echo -e "${BLUE}Group: Graph Operations${NC}"

run_test \
    "Degree command runs" \
    "$GNN_BIN degree --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Node index: 0"

run_test \
    "Neighbors command runs" \
    "$GNN_BIN neighbors --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Node index: 0"

run_test \
    "PageRank command runs" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin" \
    "PageRank"

run_test \
    "PageRank with custom damping" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin --damping=0.9" \
    "Damping factor: 0.90"

run_test \
    "PageRank with custom iterations" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin --iterations=50" \
    "Iterations: 50"

run_test \
    "In-degree command runs" \
    "$GNN_BIN in-degree --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Node index: 0"

run_test \
    "Out-degree command runs" \
    "$GNN_BIN out-degree --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Node index: 0"

echo ""

# ============================================
# Edge Operations
# ============================================

echo -e "${BLUE}Group: Edge Operations${NC}"

run_test \
    "Add-node command runs" \
    "$GNN_BIN add-node --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Add node"

run_test \
    "Add-edge command runs" \
    "$GNN_BIN add-edge --model=$TEMP_DIR/basic_gnn.bin --source=0 --target-node=1" \
    "Add edge"

run_test \
    "Remove-edge command runs" \
    "$GNN_BIN remove-edge --model=$TEMP_DIR/basic_gnn.bin --edge=0" \
    "Remove edge"

echo ""

# ============================================
# Gradient Flow
# ============================================

echo -e "${BLUE}Group: Gradient Flow${NC}"

run_test \
    "Gradient-flow command runs" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin" \
    "Gradient flow"

run_test \
    "Gradient-flow with layer" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin --layer=0" \
    "Layer: 0"

run_test \
    "Gradient-flow shows all layers" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin" \
    "Layer: all"

echo ""

# ============================================
# Error Handling
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Create missing --feature" \
    "$GNN_BIN create --hidden=16 --output=2 --mp-layers=2 --save=test.bin" \
    "Error: --feature is required"

run_test \
    "Create missing --hidden" \
    "$GNN_BIN create --feature=3 --output=2 --mp-layers=2 --save=test.bin" \
    "Error: --hidden is required"

run_test \
    "Create missing --output" \
    "$GNN_BIN create --feature=3 --hidden=16 --mp-layers=2 --save=test.bin" \
    "Error: --output is required"

run_test \
    "Create missing --mp-layers" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --save=test.bin" \
    "Error: --mp-layers is required"

run_test \
    "Create missing --save/--model" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2" \
    "Error: --model is required"

run_test \
    "Info missing --model" \
    "$GNN_BIN info" \
    "Error: --model is required"

run_test \
    "Load missing --model" \
    "$GNN_BIN load" \
    "Error: --model is required"

run_test \
    "Degree missing --node" \
    "$GNN_BIN degree --model=$TEMP_DIR/basic_gnn.bin" \
    "Error: --node is required"

run_test \
    "Neighbors missing --node" \
    "$GNN_BIN neighbors --model=$TEMP_DIR/basic_gnn.bin" \
    "Error: --node is required"

run_test \
    "Add-node missing --node" \
    "$GNN_BIN add-node --model=$TEMP_DIR/basic_gnn.bin" \
    "Error:"

run_test \
    "Add-edge missing --source" \
    "$GNN_BIN add-edge --model=$TEMP_DIR/basic_gnn.bin --target-node=1" \
    "Error:"

run_test \
    "Add-edge missing --target-node" \
    "$GNN_BIN add-edge --model=$TEMP_DIR/basic_gnn.bin --source=0" \
    "Error:"

run_test \
    "Remove-edge missing --edge" \
    "$GNN_BIN remove-edge --model=$TEMP_DIR/basic_gnn.bin" \
    "Error: --edge is required"

run_test \
    "Save missing --output" \
    "$GNN_BIN save --model=$TEMP_DIR/basic_gnn.bin" \
    "Error: --output is required"

run_test \
    "Unknown command" \
    "$GNN_BIN foobar" \
    "Unknown command:"

echo ""

# ============================================
# Cross-Implementation Tests (GNN vs Facade)
# ============================================

echo -e "${BLUE}Group: Cross-Implementation Compatibility${NC}"

run_test \
    "Facade help matches GNN structure" \
    "$FACADE_BIN help" \
    "COMMANDS:"

run_test \
    "Facade shows create command" \
    "$FACADE_BIN help" \
    "create"

run_test \
    "Facade shows predict command" \
    "$FACADE_BIN help" \
    "predict"

run_test \
    "Facade shows train command" \
    "$FACADE_BIN help" \
    "train"

run_test \
    "Facade shows pagerank command" \
    "$FACADE_BIN help" \
    "pagerank"

run_test \
    "Facade shows degree command" \
    "$FACADE_BIN help" \
    "degree"

run_test \
    "Facade shows neighbors command" \
    "$FACADE_BIN help" \
    "neighbors"

run_test \
    "Facade shows gradient-flow command" \
    "$FACADE_BIN help" \
    "gradient-flow"

echo ""

# ============================================
# Train Command (Basic)
# ============================================

echo -e "${BLUE}Group: Train Command${NC}"

$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/train_base.bin > /dev/null 2>&1

run_test \
    "Train command starts" \
    "$GNN_BIN train --model=$TEMP_DIR/train_base.bin --graph=dummy.json --save=$TEMP_DIR/train_out.bin --epochs=5 2>&1 || true" \
    "Training model"

run_test \
    "Train loads model" \
    "$GNN_BIN train --model=$TEMP_DIR/train_base.bin --graph=dummy.json --save=$TEMP_DIR/train_out2.bin --epochs=10 2>&1 || true" \
    "Model loaded"

run_test \
    "Train missing --model error" \
    "$GNN_BIN train --graph=dummy.json --save=out.bin" \
    "Error: --model is required"

run_test \
    "Train missing --graph error" \
    "$GNN_BIN train --model=$TEMP_DIR/train_base.bin --save=out.bin" \
    "Error: --graph is required"

echo ""

# ============================================
# Predict Command (Basic)
# ============================================

echo -e "${BLUE}Group: Predict Command${NC}"

run_test \
    "Predict command runs" \
    "$GNN_BIN predict --model=$TEMP_DIR/basic_gnn.bin --graph=dummy.json" \
    "Prediction:"

run_test \
    "Predict shows graph info" \
    "$GNN_BIN predict --model=$TEMP_DIR/basic_gnn.bin --graph=dummy.json" \
    "Graph nodes:"

echo ""

# ============================================
# Model Persistence Workflow
# ============================================

echo -e "${BLUE}Group: Model Persistence Workflow${NC}"

run_test \
    "Workflow: Create -> Save -> Load -> Info" \
    "$GNN_BIN create --feature=6 --hidden=48 --output=4 --mp-layers=3 --lr=0.002 --activation=leakyrelu --loss=bce --save=$TEMP_DIR/workflow1.bin && $GNN_BIN info --model=$TEMP_DIR/workflow1.bin" \
    "Feature size: 6"

run_test \
    "Workflow: Verify hidden persisted" \
    "$GNN_BIN info --model=$TEMP_DIR/workflow1.bin" \
    "Hidden size: 48"

run_test \
    "Workflow: Verify output persisted" \
    "$GNN_BIN info --model=$TEMP_DIR/workflow1.bin" \
    "Output size: 4"

run_test \
    "Workflow: Create -> Copy -> Verify identical" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/orig.bin && $GNN_BIN save --model=$TEMP_DIR/orig.bin --output=$TEMP_DIR/copy.bin && cmp -s $TEMP_DIR/orig.bin $TEMP_DIR/copy.bin && echo 'Files match'" \
    "Files match"

run_test \
    "Workflow: Multiple save operations" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/multi1.bin && $GNN_BIN save --model=$TEMP_DIR/multi1.bin --output=$TEMP_DIR/multi2.bin && $GNN_BIN save --model=$TEMP_DIR/multi2.bin --output=$TEMP_DIR/multi3.bin && cmp -s $TEMP_DIR/multi1.bin $TEMP_DIR/multi3.bin && echo 'Chain save successful'" \
    "Chain save successful"

echo ""

# ============================================
# Stress Tests
# ============================================

echo -e "${BLUE}Group: Stress Tests${NC}"

run_test \
    "Large feature size (100)" \
    "$GNN_BIN create --feature=100 --hidden=64 --output=10 --mp-layers=2 --save=$TEMP_DIR/stress1.bin" \
    "Feature size: 100"

run_test \
    "Large hidden size (256)" \
    "$GNN_BIN create --feature=10 --hidden=256 --output=10 --mp-layers=2 --save=$TEMP_DIR/stress2.bin" \
    "Hidden size: 256"

run_test \
    "Large output size (50)" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=50 --mp-layers=2 --save=$TEMP_DIR/stress3.bin" \
    "Output size: 50"

run_test \
    "Many MP layers (8)" \
    "$GNN_BIN create --feature=10 --hidden=32 --output=5 --mp-layers=8 --save=$TEMP_DIR/stress4.bin" \
    "Message passing layers: 8"

run_test \
    "Very small learning rate" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --lr=0.00001 --save=$TEMP_DIR/stress5.bin" \
    "Created GNN model"

echo ""

# ============================================
# Facade-Specific Tests
# ============================================

echo -e "${BLUE}Group: Facade-Specific Tests${NC}"

run_test \
    "Facade create command" \
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=$TEMP_DIR/facade_model.bin" \
    "Created GNN model"

run_test \
    "Facade info command" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_model.bin" \
    "Model Information"

run_test \
    "Facade pagerank command" \
    "$FACADE_BIN pagerank --model=$TEMP_DIR/facade_model.bin --damping=0.85 --iterations=20" \
    "PageRank"

run_test \
    "Facade degree command" \
    "$FACADE_BIN degree --model=$TEMP_DIR/facade_model.bin --node=0" \
    "degree"

echo ""

# ============================================
# Summary
# ============================================

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

echo "========================================="
echo "GNN CUDA Implementation Summary"
echo "========================================="
echo ""
echo "Tested Implementations:"
echo "  • gnn.cu (CUDA GNN)"
echo "  • facaded_gnn.cu (Facade Pattern GNN)"
echo ""
echo "Test Categories:"
echo "  ✓ Help & Usage"
echo "  ✓ Model Creation (various architectures)"
echo "  ✓ Hyperparameters (lr, activation, loss)"
echo "  ✓ Model Info"
echo "  ✓ Save & Load"
echo "  ✓ Graph Operations (degree, neighbors, pagerank)"
echo "  ✓ Edge Operations (add-node, add-edge, remove-edge)"
echo "  ✓ Gradient Flow Analysis"
echo "  ✓ Error Handling"
echo "  ✓ Cross-Implementation Compatibility"
echo "  ✓ Train & Predict"
echo "  ✓ Model Persistence Workflow"
echo "  ✓ Stress Tests"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
