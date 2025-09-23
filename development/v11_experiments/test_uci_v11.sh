#!/bin/bash
# Quick UCI validation test for V7P3R v11

echo "Testing V7P3R v11 UCI compliance..."
echo "===================================="

# Test basic UCI protocol
echo "Testing basic UCI protocol..."
{
    echo "uci"
    sleep 1
    echo "isready"
    sleep 1
    echo "ucinewgame"
    sleep 1
    echo "position startpos"
    sleep 1
    echo "go depth 2"
    sleep 5
    echo "quit"
} | ./dist/V7P3R_v11.exe > uci_test_output.txt 2>&1

echo "Test completed. Output saved to uci_test_output.txt"
echo ""
echo "Checking for key UCI responses..."
echo "=================================="

if grep -q "id name" uci_test_output.txt; then
    echo "✅ Engine identification found"
else
    echo "❌ Missing engine identification"
fi

if grep -q "uciok" uci_test_output.txt; then
    echo "✅ UCI acknowledgment found"
else
    echo "❌ Missing UCI acknowledgment"
fi

if grep -q "readyok" uci_test_output.txt; then
    echo "✅ Ready acknowledgment found"
else
    echo "❌ Missing ready acknowledgment"
fi

if grep -q "bestmove" uci_test_output.txt; then
    echo "✅ Best move response found"
else
    echo "❌ Missing best move response"
fi

echo ""
echo "Full output:"
echo "============"
cat uci_test_output.txt