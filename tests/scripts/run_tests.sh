#!/bin/bash
# Test Suite Runner Script
# Provides convenient commands for running different test scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_services() {
    print_header "Checking Services"
    
    # Check LocalStack
    if curl -s http://localhost:4566/_localstack/health > /dev/null 2>&1; then
        print_success "LocalStack is running"
    else
        print_warning "LocalStack is not running. Start with: make test-infra-up"
    fi
    
    # Check ChromaDB
    if curl -s http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
        print_success "ChromaDB is running"
    else
        print_warning "ChromaDB is not running. Start with: make test-infra-up"
    fi
}

generate_data() {
    print_header "Generating Test Data"
    
    if python tests/data/generate_synthetic_data.py; then
        print_success "Test data generated"
    else
        print_error "Failed to generate test data"
        exit 1
    fi
}

seed_chroma() {
    print_header "Seeding ChromaDB"
    
    if python tests/scripts/seed_chroma.py; then
        print_success "ChromaDB seeded with FAQ documents"
    else
        print_error "Failed to seed ChromaDB"
        exit 1
    fi
}

run_unit_tests() {
    print_header "Running Unit Tests"
    pytest tests/unit/ -v "$@"
}

run_integration_tests() {
    print_header "Running Integration Tests"
    pytest tests/integration/ -v -s "$@"
}

run_all_tests() {
    print_header "Running All Tests"
    pytest tests/ -v "$@"
}

run_docker_tests() {
    print_header "Running Tests in Docker"
    docker-compose -f docker-compose.test.yml up test-runner
}

run_coverage() {
    print_header "Running Tests with Coverage"
    pytest tests/ --cov=src --cov-report=html --cov-report=term "$@"
    print_success "Coverage report generated in htmlcov/"
}

show_help() {
    cat << EOF
Customer Service AI Platform - Test Suite Runner

Usage: $0 [command] [options]

Commands:
    setup           Full setup: start services, generate data, seed ChromaDB
    unit            Run unit tests only
    integration     Run integration tests only
    e2e             Run end-to-end tests only
    all             Run all tests
    docker          Run tests in Docker containers
    coverage        Run tests with coverage report
    data            Generate synthetic test data
    seed            Seed ChromaDB with FAQ documents
    check           Check if services are running
    help            Show this help message

Options:
    -v, --verbose   Verbose output
    -k EXPRESSION   Only run tests matching the given expression
    -x              Stop on first failure
    --tb=style      Traceback print mode (auto/long/short/line/native/no)

Examples:
    $0 setup                    # Full test environment setup
    $0 unit                     # Run unit tests
    $0 integration -v           # Run integration tests with verbose output
    $0 all -k classifier        # Run all tests matching "classifier"
    $0 coverage                 # Generate coverage report
    $0 docker                   # Run full test suite in Docker

Makefile Alternatives:
    make test                   # Run all tests
    make test-unit              # Run unit tests
    make test-integration       # Run integration tests
    make test-infra-up          # Start test infrastructure
    make test-setup             # Full setup

EOF
}

# Main script
main() {
    case "${1:-help}" in
        setup)
            print_header "Full Test Environment Setup"
            
            echo "Starting infrastructure..."
            docker-compose -f docker-compose.test.yml up -d localstack chromadb redis
            
            echo "Waiting for services (30s)..."
            sleep 30
            
            check_services
            generate_data
            seed_chroma
            
            print_success "Test environment is ready!"
            echo ""
            echo "Run tests with: $0 unit|integration|all"
            ;;
            
        unit)
            shift
            run_unit_tests "$@"
            ;;
            
        integration)
            shift
            run_integration_tests "$@"
            ;;
            
        e2e)
            shift
            print_header "Running End-to-End Tests"
            pytest tests/integration/test_e2e.py -v -s "$@"
            ;;
            
        all)
            shift
            run_all_tests "$@"
            ;;
            
        docker)
            run_docker_tests
            ;;
            
        coverage)
            shift
            run_coverage "$@"
            ;;
            
        data)
            generate_data
            ;;
            
        seed)
            seed_chroma
            ;;
            
        check)
            check_services
            ;;
            
        help|--help|-h)
            show_help
            ;;
            
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
