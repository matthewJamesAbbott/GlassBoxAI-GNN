#!/bin/bash

#
# Matthew Abbott 2025
# GNN Tests - Testing gnn_opencl.cpp and facaded_gnn_opencl.cpp
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output_gnn"
GNN_BIN="./gnn_opencl"
FACADE_BIN="./facaded_gnn_opencl"

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

echo "Compiling gnn_opencl.cpp..."
g++ -o gnn_opencl gnn_opencl.cpp -lOpenCL 2>&1 | grep -v "pragma message" || true

echo "Compiling facaded_gnn_opencl.cpp..."
g++ -o facaded_gnn_opencl facaded_gnn_opencl.cpp -lOpenCL 2>&1 | grep -v "pragma message" || true

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
echo "GNN Comprehensive Test Suite"
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

echo -e "${BLUE}=== GNN Binary Tests ===${NC}"
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
    "Binary file created for basic GNN" \
    "$TEMP_DIR/basic_gnn.bin"

check_file_size_nonzero \
    "Binary file has non-zero size" \
    "$TEMP_DIR/basic_gnn.bin"

run_test \
    "Create output shows feature size" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=3 --save=$TEMP_DIR/gnn_test1.bin" \
    "Feature size: 5"

run_test \
    "Create output shows hidden size" \
    "$GNN_BIN create --feature=4 --hidden=64 --output=2 --mp-layers=2 --save=$TEMP_DIR/gnn_test2.bin" \
    "Hidden size: 64"

run_test \
    "Create output shows output size" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=10 --mp-layers=2 --save=$TEMP_DIR/gnn_test3.bin" \
    "Output size: 10"

run_test \
    "Create output shows mp-layers" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=4 --save=$TEMP_DIR/gnn_test4.bin" \
    "Message passing layers: 4"

run_test \
    "Create output shows saved file" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/gnn_test5.bin" \
    "Saved to:"

echo ""

# ============================================
# Model Creation - Various Architectures
# ============================================

echo -e "${BLUE}Group: Model Creation - Various Architectures${NC}"

run_test \
    "Small GNN (8 hidden)" \
    "$GNN_BIN create --feature=2 --hidden=8 --output=1 --mp-layers=1 --save=$TEMP_DIR/small_gnn.bin" \
    "Created GNN model"

run_test \
    "Medium GNN (32 hidden, 3 layers)" \
    "$GNN_BIN create --feature=4 --hidden=32 --output=4 --mp-layers=3 --save=$TEMP_DIR/medium_gnn.bin" \
    "Created GNN model"

run_test \
    "Large GNN (128 hidden, 4 layers)" \
    "$GNN_BIN create --feature=8 --hidden=128 --output=8 --mp-layers=4 --save=$TEMP_DIR/large_gnn.bin" \
    "Created GNN model"

run_test \
    "Single output GNN" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=1 --mp-layers=2 --save=$TEMP_DIR/single_out.bin" \
    "Output size: 1"

run_test \
    "Large output GNN" \
    "$GNN_BIN create --feature=3 --hidden=32 --output=20 --mp-layers=2 --save=$TEMP_DIR/large_out.bin" \
    "Output size: 20"

run_test \
    "Single MP layer" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=1 --save=$TEMP_DIR/single_mp.bin" \
    "Message passing layers: 1"

run_test \
    "Many MP layers (5)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=5 --save=$TEMP_DIR/many_mp.bin" \
    "Message passing layers: 5"

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
    "Create with high learning rate" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --lr=0.1 --save=$TEMP_DIR/lr_high.bin" \
    "Learning rate: 0.1"

run_test \
    "Create with ReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=relu --save=$TEMP_DIR/act_relu.bin" \
    "Activation: relu"

run_test \
    "Create with LeakyReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=leakyrelu --save=$TEMP_DIR/act_lrelu.bin" \
    "Activation: leakyrelu"

run_test \
    "Create with Tanh activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=tanh --save=$TEMP_DIR/act_tanh.bin" \
    "Activation: tanh"

run_test \
    "Create with Sigmoid activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --activation=sigmoid --save=$TEMP_DIR/act_sig.bin" \
    "Activation: sigmoid"

run_test \
    "Create with MSE loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --loss=mse --save=$TEMP_DIR/loss_mse.bin" \
    "Loss function: mse"

run_test \
    "Create with BCE loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --loss=bce --save=$TEMP_DIR/loss_bce.bin" \
    "Loss function: bce"

echo ""

# ============================================
# Model Info Command
# ============================================

echo -e "${BLUE}Group: Model Info${NC}"

$GNN_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=2 --lr=0.005 --activation=tanh --loss=bce --save=$TEMP_DIR/info_test.bin > /dev/null 2>&1

run_test \
    "Info command shows feature size" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "Feature size: 4"

run_test \
    "Info command shows hidden size" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "Hidden size: 24"

run_test \
    "Info command shows output size" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "Output size: 3"

run_test \
    "Info command shows learning rate" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "Learning rate:"

run_test \
    "Info command shows activation" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "Activation:"

run_test \
    "Info command shows loss function" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "Loss function:"

run_test \
    "Info command header" \
    "$GNN_BIN info --model=$TEMP_DIR/info_test.bin" \
    "GNN Model Information"

echo ""

# ============================================
# Save and Load Commands
# ============================================

echo -e "${BLUE}Group: Save & Load Commands${NC}"

$GNN_BIN create --feature=5 --hidden=20 --output=4 --mp-layers=3 --save=$TEMP_DIR/save_test.bin > /dev/null 2>&1

run_test \
    "Save command works" \
    "$GNN_BIN save --model=$TEMP_DIR/save_test.bin --output=$TEMP_DIR/saved_copy.bin" \
    "Model saved to:"

check_file_exists \
    "Saved copy file exists" \
    "$TEMP_DIR/saved_copy.bin"

compare_files \
    "Saved copy matches original" \
    "$TEMP_DIR/save_test.bin" \
    "$TEMP_DIR/saved_copy.bin"

run_test \
    "Load command works" \
    "$GNN_BIN load --model=$TEMP_DIR/save_test.bin" \
    "Model loaded from:"

run_test \
    "Load shows feature size" \
    "$GNN_BIN load --model=$TEMP_DIR/save_test.bin" \
    "Feature size: 5"

run_test \
    "Load shows hidden size" \
    "$GNN_BIN load --model=$TEMP_DIR/save_test.bin" \
    "Hidden size: 20"

run_test \
    "Load shows output size" \
    "$GNN_BIN load --model=$TEMP_DIR/save_test.bin" \
    "Output size: 4"

echo ""

# ============================================
# Graph Operations - Degree/Neighbors/PageRank
# ============================================

echo -e "${BLUE}Group: Graph Operations${NC}"

run_test \
    "Degree command accepts parameters" \
    "$GNN_BIN degree --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Node degree information"

run_test \
    "Degree shows node index" \
    "$GNN_BIN degree --model=$TEMP_DIR/basic_gnn.bin --node=5" \
    "Node index: 5"

run_test \
    "Neighbors command accepts parameters" \
    "$GNN_BIN neighbors --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Neighbor query"

run_test \
    "Neighbors shows node index" \
    "$GNN_BIN neighbors --model=$TEMP_DIR/basic_gnn.bin --node=3" \
    "Node index: 3"

run_test \
    "PageRank command accepts parameters" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin" \
    "PageRank computation"

run_test \
    "PageRank shows damping factor" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin --damping=0.9" \
    "Damping factor: 0.90"

run_test \
    "PageRank shows iterations" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin --iterations=50" \
    "Iterations: 50"

run_test \
    "PageRank default damping" \
    "$GNN_BIN pagerank --model=$TEMP_DIR/basic_gnn.bin" \
    "Damping factor: 0.85"

echo ""

# ============================================
# Edge Operations
# ============================================

echo -e "${BLUE}Group: Edge Operations${NC}"

run_test \
    "Add-node command accepts parameters" \
    "$GNN_BIN add-node --model=$TEMP_DIR/basic_gnn.bin --node=0" \
    "Add node operation"

run_test \
    "Add-edge command accepts parameters" \
    "$GNN_BIN add-edge --model=$TEMP_DIR/basic_gnn.bin --source=0 --target-node=1" \
    "Add edge operation"

run_test \
    "Add-edge shows source/target" \
    "$GNN_BIN add-edge --model=$TEMP_DIR/basic_gnn.bin --source=2 --target-node=5" \
    "Source: 2, Target: 5"

run_test \
    "Remove-edge command accepts parameters" \
    "$GNN_BIN remove-edge --model=$TEMP_DIR/basic_gnn.bin --edge=0" \
    "Remove edge operation"

run_test \
    "Remove-edge shows edge index" \
    "$GNN_BIN remove-edge --model=$TEMP_DIR/basic_gnn.bin --edge=7" \
    "Edge index: 7"

echo ""

# ============================================
# Gradient Flow
# ============================================

echo -e "${BLUE}Group: Gradient Flow${NC}"

run_test \
    "Gradient-flow command works" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin" \
    "Gradient flow analysis"

run_test \
    "Gradient-flow shows model" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin" \
    "Model:"

run_test \
    "Gradient-flow with layer parameter" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin --layer=0" \
    "Layer: 0"

run_test \
    "Gradient-flow all layers" \
    "$GNN_BIN gradient-flow --model=$TEMP_DIR/basic_gnn.bin" \
    "Layer: all"

echo ""

# ============================================
# Error Handling
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Create missing --feature" \
    "$GNN_BIN create --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/err.bin" \
    "Error: --feature is required"

run_test \
    "Create missing --hidden" \
    "$GNN_BIN create --feature=3 --output=2 --mp-layers=2 --save=$TEMP_DIR/err.bin" \
    "Error: --hidden is required"

run_test \
    "Create missing --output" \
    "$GNN_BIN create --feature=3 --hidden=16 --mp-layers=2 --save=$TEMP_DIR/err.bin" \
    "Error: --output is required"

run_test \
    "Create missing --mp-layers" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --save=$TEMP_DIR/err.bin" \
    "Error: --mp-layers is required"

run_test \
    "Create missing --save" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2" \
    "Error: --save is required"

run_test \
    "Info missing --model" \
    "$GNN_BIN info" \
    "Error: --model is required"

run_test \
    "Degree missing --node" \
    "$GNN_BIN degree --model=$TEMP_DIR/basic_gnn.bin" \
    "Error: --node is required"

run_test \
    "Add-edge missing --source" \
    "$GNN_BIN add-edge --model=$TEMP_DIR/basic_gnn.bin --target-node=1" \
    "Error: --source and --target-node are required"

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
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=$TEMP_DIR/facade_model.json" \
    "Created model"

run_test \
    "Facade info command" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_model.json" \
    "Model Information"

run_test \
    "Facade pagerank command" \
    "$FACADE_BIN pagerank --model=$TEMP_DIR/facade_model.json --damping=0.85 --iterations=20" \
    "PageRank"

run_test \
    "Facade degree command" \
    "$FACADE_BIN degree --model=$TEMP_DIR/facade_model.json --node=0" \
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
echo "GNN Implementation Summary"
echo "========================================="
echo ""
echo "Tested Implementations:"
echo "  • gnn_opencl.cpp (OpenCL GNN)"
echo "  • facaded_gnn_opencl.cpp (Facade Pattern GNN)"
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
