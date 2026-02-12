#!/bin/bash
# Monitor memory usage of a subprocess.
# Usage: ./memory_monitor.sh <output_file> <command...>
# Example: ./memory_monitor.sh /tmp/mem.csv uv run pytest tests/test_ops.py -v

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_file> <command...>"
    echo "Example: $0 /tmp/mem.csv uv run pytest tests/test_ops.py -v"
    exit 1
fi

output_file=$1
shift

echo "Running: $@"
echo "Output: $output_file"

# Write header
echo "timestamp,rss_kb" > "$output_file"

# Run the command in background
"$@" &
cmd_pid=$!

# Sample memory until process exits
while kill -0 $cmd_pid 2>/dev/null; do
    # Get RSS of the process and all its children
    mem=$(ps -o rss= -p $cmd_pid 2>/dev/null || echo 0)
    children_mem=$(pgrep -P $cmd_pid 2>/dev/null | xargs -I{} ps -o rss= -p {} 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    total_mem=$((mem + children_mem))

    if [ "$total_mem" -gt 0 ]; then
        echo "$(date +%s),$total_mem" >> "$output_file"
        mem_mb=$(echo "scale=1; $total_mem/1024" | bc)
        echo "  $(date +%H:%M:%S): ${mem_mb} MB"
    fi
    sleep 1
done

# Wait for process to fully exit and capture exit code
wait $cmd_pid
exit_code=$?

# Print summary
samples=$(wc -l < "$output_file" | tr -d ' ')
samples=$((samples - 1))  # Subtract header

if [ $samples -gt 0 ]; then
    mems=$(tail -n +2 "$output_file" | cut -d',' -f2)
    min_kb=$(echo "$mems" | sort -n | head -1)
    max_kb=$(echo "$mems" | sort -n | tail -1)
    min_mb=$(echo "scale=1; $min_kb/1024" | bc)
    max_mb=$(echo "scale=1; $max_kb/1024" | bc)

    echo ""
    echo "Process exited with code $exit_code"
    echo "Recorded $samples samples to $output_file"
    echo "Memory: min=${min_mb} MB, max=${max_mb} MB"
fi

exit $exit_code
