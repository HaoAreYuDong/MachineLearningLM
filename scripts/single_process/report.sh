#!/bin/bash

# Metric Summarizer Script
# Summarizes all JSON files in metric data directory and generates a consolidated CSV report

# Load unified parameters
source ./parameters.sh

# Override parameters if provided via command line
METRIC_DATA_DIR_ARG=""
REPORT_DATA_DIR_ARG=""
MODEL_NAME_ARG=""

# Display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Summarizes all JSON files in metric data directory and generates a consolidated CSV report"
    echo ""
    echo "Options:"
    echo "  --metric_data_dir DIR    Metric data directory path (default: $metric_data_dir)"
    echo "  --report_data_dir DIR    Report output directory path (default: $report_data_dir)"
    echo "  --model_name NAME        Model name (default: $model_name)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --model_name minzl/toy3"
    echo "  $0 --metric_data_dir /path/to/metric --report_data_dir /path/to/reports"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metric_data_dir)
            METRIC_DATA_DIR_ARG="$2"
            shift 2
            ;;
        --report_data_dir)
            REPORT_DATA_DIR_ARG="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME_ARG="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Use command line arguments if provided, otherwise use parameters.sh values
FINAL_METRIC_DATA_DIR="${METRIC_DATA_DIR_ARG:-$metric_data_dir}"
FINAL_REPORT_DATA_DIR="${REPORT_DATA_DIR_ARG:-$report_data_dir}"
FINAL_MODEL_NAME="${MODEL_NAME_ARG:-$model_name}"

# Build Python command
PYTHON_CMD="python src/evaluation/zero_summary/metric_summarizer.py"
PYTHON_CMD="$PYTHON_CMD --metric_data_dir $FINAL_METRIC_DATA_DIR"
PYTHON_CMD="$PYTHON_CMD --report_data_dir $FINAL_REPORT_DATA_DIR"

if [[ -n "$FINAL_MODEL_NAME" ]]; then
    PYTHON_CMD="$PYTHON_CMD --model_name $FINAL_MODEL_NAME"
fi

# Display execution information
echo "🚀 Starting metric summarization..."
echo "📂 Metric directory: $FINAL_METRIC_DATA_DIR"
echo "📁 Output directory: $FINAL_REPORT_DATA_DIR"
if [[ -n "$FINAL_MODEL_NAME" ]]; then
    echo "🏷️  Model name: $FINAL_MODEL_NAME"
fi
echo "💻 Execution command: $PYTHON_CMD"
echo ""

# Execute Python script
eval $PYTHON_CMD

# Check execution result
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Metric summarization completed!"
else
    echo ""
    echo "❌ Metric summarization failed!"
    exit 1
fi