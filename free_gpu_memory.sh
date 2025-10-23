#!/bin/bash

echo "🔍 Checking GPU processes..."
nvidia-smi

echo ""
echo "⚠️  WARNING: The following processes are using GPU memory:"
echo "   Process 738826: 2.29 GiB"
echo "   Process 747131: 2.32 GiB" 
echo "   Process 464058: 10.19 GiB"
echo ""
echo "To free GPU memory, you can:"
echo "1. Kill specific processes: sudo kill -9 <PID>"
echo "2. Or restart the processes if they're your own"
echo ""
echo "Example commands:"
echo "   sudo kill -9 738826"
echo "   sudo kill -9 747131" 
echo "   sudo kill -9 464058"
echo ""
echo "⚠️  Only kill processes you own or are sure about!"
